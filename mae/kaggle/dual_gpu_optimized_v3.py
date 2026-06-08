"""
Version 3 adds saving model weights to HuggingFace
"""
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import io
from contextlib import contextmanager 
import tarfile
import logging
import json
from huggingface_hub import login as hf_login, HfApi, utils as hf_utils, hf_hub_download


hf_utils.disable_progress_bars()


# --- Logger with timestamps (configured so it actually prints inside Jupyter) ---
logger = logging.getLogger('mae_train')
logger.setLevel(logging.INFO)
logger.handlers.clear()
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(_h)
logger.propagate = False


dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = dist.get_world_size()
is_main = dist.get_rank() == 0
torch.backends.cudnn.benchmark = True

# Configurations
cfg = {
    "data": {
        "d_image": 96,
        "n_channels": 3,
        "patch_size": 8,
    },
    "model": {
        "n_encoder_blocks": 12,
        "d_enc": 384,
        "n_heads_enc": 6,
        "mlp_ratio": 4,
        "n_decoder_blocks": 8,
        "d_dec": 256,
        "n_heads_dec": 4,
    },
    "metadata": {
        "data_path" : "/kaggle/input/datasets/pratt3000/stl10-binary-files/",
        "world_size" : None, # torchrun controlled
        "lr": 1.5e-4,
        "batch_size": 4096,
        "weight_decay": 0.05,
        "momentum": [0.9, 0.95],
        "warmup_epochs": 40,
        "n_epochs": 800,
        "percent_unmasked": 0.25,
        "use_autocast": True,
        "use_grad_accumulation": True,
        "micro_batch_size" : 512,
        "load_model": True,
        "r_model_path": "/kaggle/working/checkpoint_epoch_680.pt",
        "checkpoint_every": 20,
        "save_latest_checkpoint": False,
        "plot_every": 10,
        "output_dir" : "/kaggle/working", # where loss curve & presumably checkpoint dir exists
        "checkpoint_dir" : "/kaggle/working/checkpoints",
        "is_kaggle" : True,
        "use_hf" : True,
        "hf_repo" : "flamma77/mae",
        "hf_output_dir" : "ViT-S-model-1",
        "hf_pull_model": True,
        "hf_pull_model_path" : "ViT-S-model-1/checkpoint_epoch_680.pt"
    },
}



# --- Base params (from config) ---
# Image
D_IMAGE    = cfg["data"]["d_image"]
N_CHANNELS = cfg["data"]["n_channels"]
PATCH_SIZE = cfg["data"]["patch_size"]
# Encoder
D_ENC            = cfg["model"]["d_enc"]
N_HEADS_ENC      = cfg["model"]["n_heads_enc"]
N_ENCODER_BLOCKS = cfg["model"]["n_encoder_blocks"]
MLP_RATIO        = cfg["model"]["mlp_ratio"]
# Decoder
D_DEC            = cfg["model"]["d_dec"]
N_HEADS_DEC      = cfg["model"]["n_heads_dec"]
N_DECODER_BLOCKS = cfg["model"]["n_decoder_blocks"]
# Training
DATA_PATH              = cfg["metadata"]["data_path"]
WORLD_SIZE             = cfg["metadata"]["world_size"] or world_size
LR                     = cfg["metadata"]["lr"]
BATCH_SIZE             = cfg["metadata"]["batch_size"]
WEIGHT_DECAY           = cfg["metadata"]["weight_decay"]
MOMENTUM               = cfg["metadata"]["momentum"]            # [0.9, 0.95]
WARMUP_EPOCHS          = cfg["metadata"]["warmup_epochs"]
N_EPOCHS               = cfg["metadata"]["n_epochs"]
PERCENT_UNMASKED       = cfg["metadata"]["percent_unmasked"]
USE_AUTOCAST           = cfg["metadata"]["use_autocast"]
USE_GRAD_ACCUMULATION  = cfg["metadata"]["use_grad_accumulation"]
MICRO_BATCH_SIZE       = cfg["metadata"]["micro_batch_size"]
LOAD_MODEL             = cfg["metadata"]["load_model"]
R_MODEL_PATH           = cfg["metadata"]["r_model_path"]
CHECKPOINT_EVERY       = cfg["metadata"]["checkpoint_every"] # epochs between compressed checkpoints
SAVE_LATEST_CHECKPOINT = cfg["metadata"]["save_latest_checkpoint"]
PLOT_EVERY             = cfg["metadata"]["plot_every"] # epochs between loss-curve refreshes
OUTPUT_DIR             = cfg["metadata"]["output_dir"]
CHECKPOINT_DIR         = cfg["metadata"]["checkpoint_dir"]
IS_KAGGLE              = cfg["metadata"]["is_kaggle"]
USE_HF                 = cfg["metadata"]["use_hf"]
HF_REPO                = cfg["metadata"]["hf_repo"]
HF_OUTPUT_DIR          = cfg["metadata"]["hf_output_dir"]
HF_PULL_MODEL          = cfg["metadata"]["hf_pull_model"]
HF_PULL_MODEL_PATH     = cfg["metadata"]["hf_pull_model_path"]
# --- Derived params (computed) ---
D_PATCH   = (PATCH_SIZE ** 2) * N_CHANNELS 
N_PATCHES = (D_IMAGE ** 2) // (PATCH_SIZE ** 2)
N_ROWS    = D_IMAGE // PATCH_SIZE
D_ENC_MLP = int(MLP_RATIO * D_ENC)
D_DEC_MLP = int(MLP_RATIO * D_DEC)


if is_main and USE_HF:
    # HuggingFace Authenticate
    if IS_KAGGLE:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    else:
        token = os.environ["HF_TOKEN"] # load HF_TOKEN locally
        pass
    hf_login(token=token)
    hf_api = HfApi()
    # Save JSON Configuration path
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    hf_api.upload_file(
        path_or_fileobj=config_path,
        path_in_repo=os.path.join(HF_OUTPUT_DIR, "config.json"),
        repo_id="flamma77/mae",
        repo_type="model"
    )

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(D_IMAGE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2604, 0.2566, 0.2713))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2604, 0.2566, 0.2713))
])


if is_main:
    logger.info("Loading STL10 dataset")
# Load datasets
# supervised_trainset = torchvision.datasets.STL10(
#     root=DATA_PATH, split='train', download=False, transform=train_transform
# )
# testset = torchvision.datasets.STL10(
#     root=DATA_PATH, split='test', download=False, transform=test_transform
# )
unlabeled_set = torchvision.datasets.STL10(
    root=DATA_PATH, split='unlabeled', download=False, transform=train_transform
)



# DataLoaders
micro_batch_size = MICRO_BATCH_SIZE if USE_GRAD_ACCUMULATION else BATCH_SIZE
train_sampler = DistributedSampler(unlabeled_set, shuffle=True, drop_last=True)

# supervised_trainloader = DataLoader(
#     supervised_trainset, batch_size=micro_batch_size, shuffle=True, num_workers=2
# )
# testloader = DataLoader(
#     testset, batch_size=micro_batch_size, shuffle=False, num_workers=2
# )
ssl_trainloader = DataLoader(
    unlabeled_set, 
    batch_size=micro_batch_size, 
    num_workers=3, 
    sampler=train_sampler,
    drop_last=True, 
    pin_memory=True, 
    persistent_workers=True, 
    prefetch_factor=2,
)


# --- Output locations (on Kaggle, /kaggle/working is downloadable) ---
if is_main:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def upload_file_hf(path):
    hf_api.upload_file(
        path_or_fileobj=path,
        path_in_repo=(os.path.join(HF_OUTPUT_DIR, os.path.basename(path))),
        repo_id=HF_REPO,
        repo_type="model"
    )

# --- Checkpoint helpers ---
def save_checkpoint(state, path):
    """Uncompressed save (fast). Used for the always-overwritten 'latest'."""
    torch.save(state, path)
    return path

def save_compressed_checkpoint(state, path):
    """Save then gzip-tar it; removes the .pt and returns the .tar.gz path."""
    torch.save(state, path)
    tar_path = path + '.tar.gz'
    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(path, arcname=os.path.basename(path))

    if USE_HF:
        upload_file_hf(tar_path)

    os.remove(path)
    return tar_path

def load_compressed_checkpoint(tar_path, map_location=None):
    """Load a checkpoint written by save_compressed_checkpoint."""
    with tarfile.open(tar_path, 'r:gz') as tar:
        member = tar.getmembers()[0]
        buffer = io.BytesIO(tar.extractfile(member).read())
    # weights_only=False because this is your own trusted file and it holds
    # optimizer/scheduler state; torch>=2.6 defaults to True and would choke on it.
    return torch.load(buffer, map_location=map_location, weights_only=False)
    



# _test_state = {'epoch': 0, 'model_state_dict': net.state_dict(), 'loss': 0.0}
# _tar = save_compressed_checkpoint(_test_state, f'{CHECKPOINT_DIR}/_test.pt')
# print('Wrote:', _tar, f'({os.path.getsize(_tar) / 1e6:.1f} MB)')
# _loaded = load_compressed_checkpoint(_tar, map_location='cpu')
# print('Round-trip OK - keys:', list(_loaded.keys()), '| epoch:', _loaded['epoch'])
# os.remove(_tar)


def get_pos_embeddings(D, n_patches, n_rows):
    D_pos = D // 2 # 96
    row_pos = np.zeros((n_rows, D_pos))
    col_pos = np.zeros((n_rows, D_pos))

    i = np.arange(D_pos // 2) # half sine, half cosine
    denominators = 10000 ** (2 * i / D_pos)

    for row in range(0, n_rows):
        row_pos[row, 0::2] = np.sin(row / denominators) # even inddicies
        row_pos[row, 1::2] = np.cos(row / denominators) # odd indicies

    for col in range(0, n_rows):
        col_pos[col, 0::2] = np.sin(col / denominators) # even inddicies
        col_pos[col, 1::2] = np.cos(col / denominators) # odd indicies

    pos_embeddings = np.zeros((n_patches, D))
    for row in range(0, n_rows):
        for col in range(0, n_rows):
            pos_embeddings[row*n_rows + col, :] = np.concatenate([row_pos[row, :], col_pos[col, :]])
    pos_embeddings = torch.tensor(pos_embeddings, dtype=torch.float32)
    return pos_embeddings


pos_embeddings_enc = get_pos_embeddings(D_ENC, N_PATCHES, N_ROWS)
pos_embeddings_dec = get_pos_embeddings(D_DEC, N_PATCHES, N_ROWS)



class Net(nn.Module):
    """
    Masked Autoencoder (MAE) with a Vision Transformer (ViT) encoder-decoder architecture.

    Args:
        n_encoder_blocks (int): Number of encoder transformer blocks.
        n_decoder_blocks (int): Number of decoder transformer blocks.
        patch_size (int): Side length of each square patch in pixels.
        d_image (int): Spatial dimension of the input image (assumes square, so H = W = d_image).
        patch_size (int): Side length of each square patch in pixels.
        d_patch (int): Dimensionality of each patch embedding (n_channels * patch_size^2).
        n_patches (int): Total number of patches per image.
        n_rows (int): Number of patch rows in the image grid.
        d_enc (int): Hidden embedding dimension of the encoder.
        d_enc_mlp (int): Hidden dimension of the MLP blocks in the encoder.
        d_dec (int): Hidden embedding dimension of the decoder.
        d_dec_mlp (int): Hidden dimension of the MLP blocks in the decoder.
        n_heads_enc (int): Number of attention heads per encoder transformer block.
        n_heads_dec (int): Number of attention heads per decoder transformer block.
        pos_embeddings_enc (torch.Tensor): Positional embeddings for the encoder,
            shape ``(1, n_patches, D)``.
        pos_embedding_dec (torch.Tensor): Positional embeddings for the decoder,
            shape ``(1, n_patches, D_decoder)``.
        percent_unmasked (float): Fraction of patch embeddings visible to the encoder,
            in the range ``(0, 1]``.
    """
    def __init__(
        self,
        n_encoder_blocks,
        n_decoder_blocks,
        d_image,
        patch_size,
        d_patch,
        n_patches,
        n_rows,
        d_enc,
        d_enc_mlp,
        d_dec,
        d_dec_mlp,
        n_heads_enc,
        n_heads_dec,
        pos_embeddings_enc,
        pos_embeddings_dec,
        percent_unmasked,
    ):
        super(Net, self).__init__()

        self.d_image = d_image
        self.patch_size = patch_size
        self.d_patch = d_patch
        self.n_patches = n_patches
        self.n_rows = n_rows
        self.d_enc = d_enc
        self.d_enc_mlp = d_enc_mlp
        self.d_dec = d_dec
        self.d_dec_mlp = d_dec_mlp
        self.register_buffer("pos_embeddings_enc", pos_embeddings_enc)
        self.register_buffer("pos_embeddings_dec", pos_embeddings_dec)
        self.percent_unmasked = percent_unmasked

        ### ENCODER
        self.img2enc_projection = nn.Linear(d_patch, d_enc)
        self.encoder_blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm_a' : nn.LayerNorm(d_enc),
                'msa' : nn.MultiheadAttention(d_enc, n_heads_enc, batch_first=True),
                'norm_b': nn.LayerNorm(d_enc),
                'mlp_a': nn.Linear(d_enc, d_enc_mlp),
                'mlp_b': nn.Linear(d_enc_mlp, d_enc)
            })
            for _ in range(n_encoder_blocks)
        ])

        ### DECODER (just single transformer block)
        self.masked_embedding = nn.Parameter(torch.randn(1, 1, d_dec))
        self.enc2dec_projection = nn.Linear(d_enc, d_dec)
        self.decoder_blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm_a' : nn.LayerNorm(d_dec),
                'msa' : nn.MultiheadAttention(d_dec, n_heads_dec, batch_first=True),
                'norm_b': nn.LayerNorm(d_dec),
                'mlp_a': nn.Linear(d_dec, d_dec_mlp),
                'mlp_b': nn.Linear(d_dec_mlp, d_dec)
            })
            for _ in range(n_decoder_blocks)
        ])
        self.dec2img_projection = nn.Linear(d_dec, d_patch)

    def encode(self, x):
        # x is (B, C, H, W) = (B, 3, 96, 96)
        B, C, H, W = x.shape

        # Make (B, 3, 12, 12, 8, 8)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # assert patches.shape == (B, C, self.n_rows, self.n_rows, self.patch_size, self.patch_size), patches.shape 
        
        # Make (B, 12, 12, 3, 8, 8)
        x = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # assert x.shape == (B, self.n_rows, self.n_rows, C, self.patch_size, self.patch_size), x.shape

        SEQ = self.n_patches # N_ROWS ** 2

        x = x.view(B, SEQ, self.d_patch)
        # assert x.shape == (B, 144, 192), x.shape
        truth_patches = x

        # Create a mask and apply it
        noise = torch.rand(B, SEQ, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1) # Get a list of sorted ids for each B
        n_keep = int(SEQ * self.percent_unmasked)
        ids_unmasked = ids_shuffle[:, :n_keep] # Store the ids of first 25% -> (B, n_keep)
        ids_masked = ids_shuffle[:, n_keep:] # Store the ids of last 75% -> (B, n_masked)

        # x_unmasked we want (B, n_keep, 192)
        ind_unmasked_enc = ids_unmasked.unsqueeze(-1).expand(-1, -1, self.d_patch) # (B, n_keep, d_patch)
        
        # For dim=1, keep all dimensions the same but replaces with the column index
        x_unmasked = torch.gather(x, 1, ind_unmasked_enc)
        # assert x_unmasked.shape == (B, n_keep, 192), x_unmasked.shape

        embeddings = self.img2enc_projection(x_unmasked)

        # Add our positional embeddings
        # (N_PATCHES, D)
        # embeddings is (B, n_keep, D)
        ind_unmasked_pos = ids_unmasked.unsqueeze(-1).expand(-1, -1, self.d_enc)
        local_pos_embeddings = torch.gather(self.pos_embeddings_enc.unsqueeze(0).expand(B, -1, -1), 1, ind_unmasked_pos)
        embeddings = embeddings + local_pos_embeddings

        # Run through all Encoder Blocks
        for block in self.encoder_blocks:
            x = block['norm_a'](embeddings)
            x, _ = block['msa'](x, x, x, need_weights=False)
            embeddings = embeddings + x # skip connection
            x = block['norm_b'](embeddings)
            x = F.gelu( block['mlp_a'](x) )
            x = block['mlp_b'](x)
            embeddings = embeddings + x # skip connection

        return embeddings, truth_patches, ids_unmasked, ids_masked

        # Ok now we need to return the meal across the embeddings
        # return embeddings.mean(dim=1) # (B, D)

    def forward(self, x):
        B, C, H, W = x.shape
        embeddings, truth_patches, ids_unmasked, ids_masked = self.encode(x)
        ind_unmasked_dec = ids_unmasked.unsqueeze(-1).expand(-1, -1, self.d_dec) # (B, n_keep, D)

        ## DECODER
        x = self.masked_embedding.repeat(B, self.n_patches, 1)
        # assert x.shape == (B, self.n_patches, self.D_decoder), x.shape
        unmasked_embeddings_dec = self.enc2dec_projection(embeddings).to(x.dtype) # cast to support autocast fp16
        embeddings_dec = x.clone().scatter(1, ind_unmasked_dec, unmasked_embeddings_dec)
        # embeddings_dec = torch.scatter(x, 1, ind_unmasked_dec, unmasked_embeddings_dec)
        embeddings_dec = embeddings_dec + self.pos_embeddings_dec # broadcast along B in pos_embeddings

        for block in self.decoder_blocks:
            x = block['norm_a'](embeddings_dec)
            x, _ = block['msa'](x, x, x, need_weights=False)
            embeddings_dec = embeddings_dec + x
            x = block['norm_b'](embeddings_dec)
            x = F.gelu( block['mlp_a'](x) )
            x = block['mlp_b'](x)
            embeddings_dec = embeddings_dec + x

        y_patches = self.dec2img_projection(embeddings_dec)

        return y_patches, truth_patches, ids_masked
    


raw_model = Net(
    n_encoder_blocks=N_ENCODER_BLOCKS,
    n_decoder_blocks=N_DECODER_BLOCKS,
    d_image=D_IMAGE,
    patch_size=PATCH_SIZE,
    d_patch=D_PATCH,
    n_patches=N_PATCHES,
    n_rows=N_ROWS,
    d_enc=D_ENC,
    d_enc_mlp=D_ENC_MLP,
    d_dec=D_DEC,
    d_dec_mlp=D_DEC_MLP,
    n_heads_enc=N_HEADS_ENC,
    n_heads_dec=N_HEADS_DEC,
    pos_embeddings_enc=pos_embeddings_enc,
    pos_embeddings_dec=pos_embeddings_dec,
    percent_unmasked=PERCENT_UNMASKED)
raw_model.to(device)

@contextmanager
def wait_for_hf_model_pull():
    if not is_main:
        dist.barrier()
    yield
    if is_main:
        dist.barrier()

criterion = nn.MSELoss()
optimizer = optim.AdamW(
    raw_model.parameters(),
    lr=LR,
    betas=tuple(MOMENTUM),   # MOMENTUM is the [0.9, 0.95] list -> tuple
    eps=1e-8,                # no config var for eps, kept as before
    weight_decay=WEIGHT_DECAY,
)
scaler = torch.amp.GradScaler('cuda', enabled=USE_AUTOCAST)

warmup = LinearLR(optimizer, start_factor=1/WARMUP_EPOCHS, total_iters=WARMUP_EPOCHS)
cosine = CosineAnnealingLR(optimizer, N_EPOCHS - WARMUP_EPOCHS)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])

if LOAD_MODEL:
    if HF_PULL_MODEL:
        with wait_for_hf_model_pull():
            if is_main:
                path = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=HF_PULL_MODEL_PATH,
                    local_dir=os.path.dirname(R_MODEL_PATH)
                )
                os.rename(path, R_MODEL_PATH)
                logger.info(f"Downloaded to: {R_MODEL_PATH}")
                logger.info("Size: {:.1f} MB".format(os.path.getsize(R_MODEL_PATH) / 1e6))
    state = torch.load(R_MODEL_PATH, map_location=device, weights_only=False)

    raw_model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    scheduler.load_state_dict(state['scheduler_state_dict'])
    scaler.load_state_dict(state['scaler_state_dict'])
    start_epoch = state['epoch']

    if is_main:
        logger.info(f"Resumed from {R_MODEL_PATH} at epoch {start_epoch} "
                    f"(saved loss {state['loss']:.4f})")

else:
    start_epoch = 0
    

ddp_model = DDP(raw_model)
model = torch.compile(ddp_model)

if is_main:
    logger.info(f"Using DistributedDataParallel across {WORLD_SIZE} GPUs. Starting training")
    logger.info(f"Running version 2")


ACCUM_STEPS = BATCH_SIZE // (MICRO_BATCH_SIZE * WORLD_SIZE) if USE_GRAD_ACCUMULATION else 1

train_losses = []
for epoch in range(start_epoch, N_EPOCHS):
    epoch_loss = torch.zeros((), device=device)
    accum_count = 0
    optimizer.zero_grad()
    for i, data in enumerate(ssl_trainloader):
        inputs = data[0].to(device, non_blocking=True)
        is_step = (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(ssl_trainloader)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AUTOCAST):
            y_pred_patches, y_true_patches, ids_masked = model(inputs)
            masked_indices = ids_masked.unsqueeze(-1).expand(-1, -1, D_PATCH)
            y_pred_masked_patches = torch.gather(y_pred_patches, 1, masked_indices)
    
            with torch.no_grad():
                y_true_patches = torch.gather(y_true_patches, 1, masked_indices)
                # normalize each patch to zero mean, unit variance
                mean = y_true_patches.mean(dim=-1, keepdim=True)
                var = y_true_patches.var(dim=-1, keepdim=True)
                y_true_patches = (y_true_patches - mean) / (var + 1e-6).sqrt()
    
            loss = criterion(y_pred_masked_patches, y_true_patches)
            
        epoch_loss += loss.detach()
        accum_count += 1

        if is_step:
            scaler.scale(loss / ACCUM_STEPS).backward() # trigger DDP all-reduce with gradient_buckets
            scaler.step(optimizer) # will function as identity when USE_AUTOCAST false
            scaler.update()
            optimizer.zero_grad()
            running_loss = torch.zeros((), device=device)
            accum_count = 0
        else:
            with ddp_model.no_sync():
                scaler.scale(loss / ACCUM_STEPS).backward()
        
    scheduler.step()
    dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
    avg_loss = (epoch_loss / (len(ssl_trainloader) * WORLD_SIZE)).item()
    
    if is_main:
        train_losses.append(avg_loss)
        logger.info(f'Epoch {epoch + 1}/{N_EPOCHS} - loss: {avg_loss:.6f}')

        if SAVE_LATEST_CHECKPOINT:
            # always-overwritten 'latest' for crash recovery (uncompressed, fast)
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict' : scaler.state_dict(),
                'loss': avg_loss,
            }, f'{CHECKPOINT_DIR}/latest.pt')

        # refresh the downloadable loss curve
        if (epoch + 1) % PLOT_EVERY == 0:
            plt.figure(figsize=(14, 4))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses)
            plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
            plt.title("MAE Training Loss (Full)")
            plt.yscale("log")
            plt.subplot(1, 2, 2)
            last_n = min(10, len(train_losses))
            plt.plot(range(len(train_losses) - last_n, len(train_losses)), train_losses[-last_n:])
            plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
            plt.title("MAE Training Loss (Last 10 Epochs)")
            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
            plt.savefig(plot_path)
            plt.close()

            if USE_HF:
                upload_file_hf(plot_path)

        # periodic compressed snapshot for history
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            path = save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict' : scaler.state_dict(),
                'loss': avg_loss,
            }, f'{CHECKPOINT_DIR}/checkpoint_epoch_{epoch + 1}.pt')
            logger.info(f'Checkpoint saved: {path}')

    dist.barrier() # keep ranks in step before continuining

dist.destroy_process_group()

if is_main:
    logger.info('Finished training')


