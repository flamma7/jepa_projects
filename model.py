import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        self.enc_terminal_norm = nn.LayerNorm(d_enc)

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
        self.dec_terminal_norm = nn.LayerNorm(d_dec)
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

        embeddings = self.enc_terminal_norm(embeddings)
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

        embeddings_dec = self.dec_terminal_norm(embeddings_dec)
        y_patches = self.dec2img_projection(embeddings_dec)

        return y_patches, truth_patches, ids_masked
