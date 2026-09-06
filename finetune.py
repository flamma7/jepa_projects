"""
Evaluate MAE checkpoints with a linear probe, an MLP probe, and last-block
fine-tuning. After each method, val/test accuracy is written to results.npz
so a run can resume after interruption.
"""
import json
import os
import re

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, random_split

from model import Net, get_pos_embeddings


# --- Checkpoints to evaluate ---
MODEL_DIR = "checkpoints"
CHECKPOINTS = [
    "checkpoint_epoch_100.pt",
    "checkpoint_epoch_400.pt",
    "checkpoint_epoch_800.pt",
    "checkpoint_epoch_1600.pt"
]
RESULTS_PATH = "results.npz"

METHODS = ("linear", "mlp", "last_block")

SEED = 12
BATCH_SIZE = 128
N_CLASSES = 10
N_EPOCHS = 50

# Linear probe
LINEAR_LR = 3e-3

# MLP probe
MLP_LR = 1e-3
MLP_DIM = 512
MLP_WARMUP_EPOCHS = 10
MLP_WEIGHT_DECAY = 1e-4
MLP_DROPOUT = 0.1

# Last transformer block
LAST_BLOCK_LR = 1e-4
LAST_BLOCK_WARMUP_EPOCHS = 5
LAST_BLOCK_WEIGHT_DECAY = 0.05


def epoch_from_checkpoint(name):
    match = re.search(r"(\d+)", name)
    if match is None:
        raise ValueError(f"Could not parse epoch from checkpoint name: {name}")
    return int(match.group(1))


def load_config():
    with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
        return json.load(f)


def build_model(cfg, device):
    d_image = cfg["data"]["d_image"]
    n_channels = cfg["data"]["n_channels"]
    patch_size = cfg["data"]["patch_size"]
    d_enc = cfg["model"]["d_enc"]
    n_heads_enc = cfg["model"]["n_heads_enc"]
    n_encoder_blocks = cfg["model"]["n_encoder_blocks"]
    mlp_ratio = cfg["model"]["mlp_ratio"]
    d_dec = cfg["model"]["d_dec"]
    n_heads_dec = cfg["model"]["n_heads_dec"]
    n_decoder_blocks = cfg["model"]["n_decoder_blocks"]
    percent_unmasked = cfg["metadata"]["percent_unmasked"]

    d_patch = (patch_size ** 2) * n_channels
    n_patches = (d_image ** 2) // (patch_size ** 2)
    n_rows = d_image // patch_size
    d_enc_mlp = int(mlp_ratio * d_enc)
    d_dec_mlp = int(mlp_ratio * d_dec)

    pos_embeddings_enc = get_pos_embeddings(d_enc, n_patches, n_rows)
    pos_embeddings_dec = get_pos_embeddings(d_dec, n_patches, n_rows)
    net = Net(
        n_encoder_blocks=n_encoder_blocks,
        n_decoder_blocks=n_decoder_blocks,
        d_image=d_image,
        patch_size=patch_size,
        d_patch=d_patch,
        n_patches=n_patches,
        n_rows=n_rows,
        d_enc=d_enc,
        d_enc_mlp=d_enc_mlp,
        d_dec=d_dec,
        d_dec_mlp=d_dec_mlp,
        n_heads_enc=n_heads_enc,
        n_heads_dec=n_heads_dec,
        pos_embeddings_enc=pos_embeddings_enc,
        pos_embeddings_dec=pos_embeddings_dec,
        percent_unmasked=percent_unmasked,
    )
    net.to(device)
    return net, d_enc, d_image


def load_checkpoint(net, checkpoint_name, device):
    path = os.path.join(MODEL_DIR, checkpoint_name)
    sd = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
    net.load_state_dict(sd)
    net.to(device)
    for p in net.parameters():
        p.requires_grad = False
    net.eval()


def make_dataloaders(d_image):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(d_image, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2604, 0.2566, 0.2713)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
    ])
    supervised_trainset = torchvision.datasets.STL10(
        root="./data", split="train", download=True, transform=train_transform
    )
    testset = torchvision.datasets.STL10(
        root="./data", split="test", download=True, transform=test_transform
    )
    val_size = 500
    train_size = len(supervised_trainset) - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_subset, val_subset = random_split(
        supervised_trainset, [train_size, val_size], generator=generator
    )
    trainloader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    valloader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    testloader = DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    return trainloader, valloader, testloader


def encode_mean(net, inputs):
    embeddings, _, _, _ = net.encode(inputs, mask=False)
    return embeddings.mean(dim=1)


@torch.no_grad()
def evaluate(net, head, loader, device, train_encoder=False):
    head.eval()
    if train_encoder:
        net.eval()
    correct = 0
    total = 0
    for data in loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        reps = encode_mean(net, inputs)
        preds = head(reps).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def init_results(epochs):
    n_ckpts = len(epochs)
    n_methods = len(METHODS)
    if os.path.exists(RESULTS_PATH):
        data = np.load(RESULTS_PATH, allow_pickle=False)
        saved_epochs = data["epochs"].astype(int)
        saved_methods = [str(m) for m in data["methods"]]
        if np.array_equal(saved_epochs, np.asarray(epochs)) and saved_methods == list(METHODS):
            print(f"Loaded existing results from {RESULTS_PATH}")
            return {
                "epochs": saved_epochs,
                "methods": np.array(METHODS),
                "val_acc": data["val_acc"].astype(np.float64),
                "test_acc": data["test_acc"].astype(np.float64),
            }
        print(f"Existing {RESULTS_PATH} does not match this run; starting fresh")
    return {
        "epochs": np.asarray(epochs, dtype=int),
        "methods": np.array(METHODS),
        "val_acc": np.full((n_ckpts, n_methods), np.nan),
        "test_acc": np.full((n_ckpts, n_methods), np.nan),
    }


def save_results(results):
    np.savez(RESULTS_PATH, **results)
    print(f"Saved progress to {RESULTS_PATH}")


def is_done(results, ckpt_idx, method):
    j = METHODS.index(method)
    return np.isfinite(results["test_acc"][ckpt_idx, j])


def record(results, ckpt_idx, method, val_acc, test_acc):
    j = METHODS.index(method)
    results["val_acc"][ckpt_idx, j] = val_acc
    results["test_acc"][ckpt_idx, j] = test_acc
    save_results(results)


def train_linear_probe(net, trainloader, valloader, testloader, device, d_enc):
    net.eval()
    linear_probe = nn.Linear(d_enc, N_CLASSES).to(device)
    optimizer = optim.AdamW(linear_probe.parameters(), lr=LINEAR_LR, weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    best_epoch = -1

    for epoch in range(N_EPOCHS):
        linear_probe.train()
        running_loss = 0.0
        n_batches = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            with torch.no_grad():
                reps = encode_mean(net, inputs)
            logits = linear_probe(reps)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i + 1):.6f}")

        scheduler.step()
        train_loss = running_loss / n_batches
        val_acc = evaluate(net, linear_probe, valloader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch + 1:3d} | train_loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f} | lr={current_lr:.2e}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in linear_probe.state_dict().items()}

    linear_probe.load_state_dict(best_state)
    test_acc = evaluate(net, linear_probe, testloader, device)
    print(f"Finished linear probe. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Test accuracy: {test_acc:.4f}")
    return best_val_acc, test_acc


def train_mlp_probe(net, trainloader, valloader, testloader, device, d_enc):
    net.eval()
    mlp_probe = nn.Sequential(
        nn.LayerNorm(d_enc),
        nn.Linear(d_enc, MLP_DIM),
        nn.GELU(),
        nn.Dropout(MLP_DROPOUT),
        nn.Linear(MLP_DIM, N_CLASSES),
    ).to(device)
    optimizer = optim.AdamW(mlp_probe.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=MLP_WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS - MLP_WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[MLP_WARMUP_EPOCHS])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    best_epoch = -1

    for epoch in range(N_EPOCHS):
        mlp_probe.train()
        running_loss = 0.0
        n_batches = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            with torch.no_grad():
                reps = encode_mean(net, inputs)
            logits = mlp_probe(reps)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i + 1):.6f}")

        scheduler.step()
        train_loss = running_loss / n_batches
        val_acc = evaluate(net, mlp_probe, valloader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch + 1:3d} | train_loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f} | lr={current_lr:.2e}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in mlp_probe.state_dict().items()}

    mlp_probe.load_state_dict(best_state)
    test_acc = evaluate(net, mlp_probe, testloader, device)
    print(f"Finished MLP probe. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Test accuracy: {test_acc:.4f}")
    return best_val_acc, test_acc


def train_last_block(net, trainloader, valloader, testloader, device, d_enc):
    for p in net.parameters():
        p.requires_grad = False

    last_block = net.encoder_blocks[-1]
    for p in last_block.parameters():
        p.requires_grad = True
    for p in net.enc_terminal_norm.parameters():
        p.requires_grad = True

    trainable = [n for n, p in net.named_parameters() if p.requires_grad]
    print(f"trainable encoder tensors ({len(trainable)}):")
    for n in trainable:
        print(f"  {n}")

    cls_head = nn.Linear(d_enc, N_CLASSES).to(device)
    trainable_params = (
        list(last_block.parameters())
        + list(net.enc_terminal_norm.parameters())
        + list(cls_head.parameters())
    )
    optimizer = optim.AdamW(
        trainable_params,
        lr=LAST_BLOCK_LR,
        betas=(0.9, 0.999),
        weight_decay=LAST_BLOCK_WEIGHT_DECAY,
    )
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=LAST_BLOCK_WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS - LAST_BLOCK_WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer, [warmup, cosine], milestones=[LAST_BLOCK_WARMUP_EPOCHS]
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    best_epoch = -1

    for epoch in range(N_EPOCHS):
        last_block.train()
        net.enc_terminal_norm.train()
        cls_head.train()
        running_loss = 0.0
        n_batches = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            reps = encode_mean(net, inputs)
            logits = cls_head(reps)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i + 1):.6f}")

        scheduler.step()
        train_loss = running_loss / n_batches
        last_block.eval()
        net.enc_terminal_norm.eval()
        cls_head.eval()
        val_acc = evaluate(net, cls_head, valloader, device, train_encoder=True)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch + 1:3d} | train_loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f} | lr={current_lr:.2e}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {
                "last_block": {
                    k: v.detach().clone() for k, v in last_block.state_dict().items()
                },
                "enc_terminal_norm": {
                    k: v.detach().clone()
                    for k, v in net.enc_terminal_norm.state_dict().items()
                },
                "cls_head": {
                    k: v.detach().clone() for k, v in cls_head.state_dict().items()
                },
            }

    last_block.load_state_dict(best_state["last_block"])
    net.enc_terminal_norm.load_state_dict(best_state["enc_terminal_norm"])
    cls_head.load_state_dict(best_state["cls_head"])
    test_acc = evaluate(net, cls_head, testloader, device, train_encoder=True)
    print(f"Finished last-block finetune. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Test accuracy: {test_acc:.4f}")
    return best_val_acc, test_acc


def format_cell(value):
    return f"{value:.4f}" if np.isfinite(value) else "—"


def print_table(results, acc_key, title):
    epochs = results["epochs"]
    methods = [str(m) for m in results["methods"]]
    acc = results[acc_key]
    col_w = max(12, max(len(m) for m in methods) + 2)
    header = f"{'epoch':>8}" + "".join(f"{m:>{col_w}}" for m in methods)
    print()
    print(title)
    print(header)
    print("-" * len(header))
    for i, ep in enumerate(epochs):
        row = f"{int(ep):>8}"
        for j in range(len(methods)):
            row += f"{format_cell(acc[i, j]):>{col_w}}"
        print(row)


def main():
    device = torch.device("cuda:0")
    cfg = load_config()
    net, d_enc, d_image = build_model(cfg, device)
    trainloader, valloader, testloader = make_dataloaders(d_image)

    epochs = [epoch_from_checkpoint(name) for name in CHECKPOINTS]
    results = init_results(epochs)

    trainers = {
        "linear": train_linear_probe,
        "mlp": train_mlp_probe,
        "last_block": train_last_block,
    }

    for ckpt_idx, checkpoint_name in enumerate(CHECKPOINTS):
        epoch = epochs[ckpt_idx]
        print(f"\n{'=' * 60}")
        print(f"Checkpoint {checkpoint_name} (pretrain epoch {epoch})")
        print("=" * 60)

        for method, trainer in trainers.items():
            if is_done(results, ckpt_idx, method):
                print(f"Skipping {method} for epoch {epoch} (already in {RESULTS_PATH})")
                continue

            print(f"\n--- {method} | epoch {epoch} ---")
            load_checkpoint(net, checkpoint_name, device)
            val_acc, test_acc = trainer(
                net, trainloader, valloader, testloader, device, d_enc
            )
            record(results, ckpt_idx, method, val_acc, test_acc)

    print_table(results, "test_acc", "Test accuracy")
    print_table(results, "val_acc", "Validation accuracy")


if __name__ == "__main__":
    main()
