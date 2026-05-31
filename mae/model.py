import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_pos_embeddings(D, n_patches, n_rows, device):
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
    pos_embeddings = pos_embeddings.to(device)
    return pos_embeddings

class Net(nn.Module):
    """
    Masked Autoencoder (MAE) with a Vision Transformer (ViT) encoder-decoder architecture.

    Args:
        D_image (int): Spatial dimension of the input image (assumes square, so H = W = D_image).
        patch_size (int): Side length of each square patch in pixels.
        D_patch (int): Dimensionality of each patch embedding (n_channels * patch_size^2).
        n_patches (int): Total number of patches per image.
        n_rows (int): Number of patch rows in the image grid.
        D (int): Hidden embedding dimension of the encoder.
        D_mlp (int): Hidden dimension of the MLP blocks in the encoder.
        D_decoder (int): Hidden embedding dimension of the decoder.
        D_decoder_mlp (int): Hidden dimension of the MLP blocks in the decoder.
        n_heads (int): Number of attention heads per transformer block.
        pos_embeddings_enc (torch.Tensor): Positional embeddings for the encoder,
            shape ``(1, n_patches, D)``.
        pos_embedding_dec (torch.Tensor): Positional embeddings for the decoder,
            shape ``(1, n_patches, D_decoder)``.
        percent_unmasked (float): Fraction of patch embeddings visible to the encoder,
            in the range ``(0, 1]``.
    """
    def __init__(
        self,
        D_image,
        patch_size,
        D_patch,
        n_patches,
        n_rows,
        D,
        D_mlp,
        D_decoder,
        D_decoder_mlp,
        n_heads,
        pos_embeddings_enc,
        pos_embedding_dec,
        percent_unmasked,
    ):
        super(Net, self).__init__()

        self.D_image = D_image
        self.patch_size = patch_size
        self.D_patch = D_patch
        self.n_patches = n_patches
        self.n_rows = n_rows
        self.D = D
        self.D_mlp = D_mlp
        self.D_decoder = D_decoder
        self.D_decoder_mlp = D_decoder_mlp
        self.pos_embeddings_enc = pos_embeddings_enc
        self.pos_embeddings_dec = pos_embedding_dec
        self.percent_unmasked = percent_unmasked


        ### ENCODER

        self.img2enc_projection = nn.Linear(D_patch, D)

        # Block 1
        self.norm1a = nn.LayerNorm(D)
        self.msa1 = nn.MultiheadAttention(D, n_heads, batch_first=True)
        self.norm1b = nn.LayerNorm(D)
        self.mlp1a = nn.Linear(D, D_mlp)
        self.mlp1b = nn.Linear(D_mlp, D)

        # Block 2
        self.norm2a = nn.LayerNorm(D)
        self.msa2 = nn.MultiheadAttention(D, n_heads, batch_first=True)
        self.norm2b = nn.LayerNorm(D)
        self.mlp2a = nn.Linear(D, D_mlp)
        self.mlp2b = nn.Linear(D_mlp, D)

        # Block 3
        self.norm3a = nn.LayerNorm(D)
        self.msa3 = nn.MultiheadAttention(D, n_heads, batch_first=True)
        self.norm3b = nn.LayerNorm(D)
        self.mlp3a = nn.Linear(D, D_mlp)
        self.mlp3b = nn.Linear(D_mlp, D)

        # Block 4
        self.norm4a = nn.LayerNorm(D)
        self.msa4 = nn.MultiheadAttention(D, n_heads, batch_first=True)
        self.norm4b = nn.LayerNorm(D)
        self.mlp4a = nn.Linear(D, D_mlp)
        self.mlp4b = nn.Linear(D_mlp, D)

        # Block 5
        self.norm5a = nn.LayerNorm(D)
        self.msa5 = nn.MultiheadAttention(D, n_heads, batch_first=True)
        self.norm5b = nn.LayerNorm(D)
        self.mlp5a = nn.Linear(D, D_mlp)
        self.mlp5b = nn.Linear(D_mlp, D)

        # Block 6
        self.norm6a = nn.LayerNorm(D)
        self.msa6 = nn.MultiheadAttention(D, n_heads, batch_first=True)
        self.norm6b = nn.LayerNorm(D)
        self.mlp6a = nn.Linear(D, D_mlp)
        self.mlp6b = nn.Linear(D_mlp, D)

        ### DECODER (just single transformer block)
        self.masked_embedding = nn.Parameter(torch.randn(1, 1, D_decoder))
        self.enc2dec_projection = nn.Linear(D, D_decoder)
        self.decoder_norm1 = nn.LayerNorm(D_decoder)
        self.decoder_msa = nn.MultiheadAttention(D_decoder, n_heads, batch_first=True)
        self.decoder_norm2 = nn.LayerNorm(D_decoder)
        self.decoder_mlp1 = nn.Linear(D_decoder, D_decoder_mlp)
        self.decoder_mlp2 = nn.Linear(D_decoder_mlp, D_decoder)
        self.dec2img_projection = nn.Linear(D_decoder, D_patch)

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

        x = x.view(B, SEQ, self.D_patch)
        # assert x.shape == (B, 144, 192), x.shape
        truth_patches = x

        # Create a mask and apply it
        noise = torch.rand(B, SEQ, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1) # Get a list of sorted ids for each B
        n_keep = int(SEQ * self.percent_unmasked)
        ids_unmasked = ids_shuffle[:, :n_keep] # Store the ids of first 25% -> (B, n_keep)
        ids_masked = ids_shuffle[:, n_keep:] # Store the ids of last 75% -> (B, n_masked)

        # x_unmasked we want (B, n_keep, 192)
        ind_unmasked_enc = ids_unmasked.unsqueeze(-1).expand(-1, -1, self.D_patch) # (B, n_keep, D_patch)
        
        # For dim=1, keep all dimensions the same but replaces with the column index
        x_unmasked = torch.gather(x, 1, ind_unmasked_enc)
        # assert x_unmasked.shape == (B, n_keep, 192), x_unmasked.shape

        embeddings = self.img2enc_projection(x_unmasked)

        # Add our positional embeddings
        # (N_PATCHES, D)
        # embeddings is (B, n_keep, D)
        ind_unmasked_pos = ids_unmasked.unsqueeze(-1).expand(-1, -1, self.D)
        local_pos_embeddings = torch.gather(self.pos_embeddings_enc.unsqueeze(0).expand(B, -1, -1), 1, ind_unmasked_pos)
        embeddings = embeddings + local_pos_embeddings

        # Block 1
        x = self.norm1a(embeddings)
        x, _ = self.msa1(x, x, x)
        embeddings = embeddings + x # skip connection
        x = self.norm1b(embeddings)
        x = F.gelu( self.mlp1a(x) )
        x = self.mlp1b(x)
        embeddings = embeddings + x # skip connection
        # Block 2
        x = self.norm2a(embeddings)
        x, _ = self.msa2(x, x, x)
        embeddings = embeddings + x # skip connection
        x = self.norm2b(embeddings)
        x = F.gelu( self.mlp2a(x) )
        x = self.mlp2b(x)
        embeddings = embeddings + x # skip connection
        # Block 3
        x = self.norm3a(embeddings)
        x, _ = self.msa3(x, x, x)
        embeddings = embeddings + x # skip connection
        x = self.norm3b(embeddings)
        x = F.gelu( self.mlp3a(x) )
        x = self.mlp3b(x)
        embeddings = embeddings + x # skip connection
        # Block 4
        x = self.norm4a(embeddings)
        x, _ = self.msa4(x, x, x)
        embeddings = embeddings + x # skip connection
        x = self.norm4b(embeddings)
        x = F.gelu( self.mlp4a(x) )
        x = self.mlp4b(x)
        embeddings = embeddings + x # skip connection
        # Block 5
        x = self.norm5a(embeddings)
        x, _ = self.msa5(x, x, x)
        embeddings = embeddings + x # skip connection
        x = self.norm5b(embeddings)
        x = F.gelu( self.mlp5a(x) )
        x = self.mlp5b(x)
        embeddings = embeddings + x # skip connection
        # Block 6
        x = self.norm6a(embeddings)
        x, _ = self.msa6(x, x, x)
        embeddings = embeddings + x # skip connection
        x = self.norm6b(embeddings)
        x = F.gelu( self.mlp6a(x) )
        x = self.mlp6b(x)
        embeddings = embeddings + x # skip connection

        return embeddings, truth_patches, ids_unmasked, ids_masked

        # Ok now we need to return the meal across the embeddings
        # return embeddings.mean(dim=1) # (B, D)

    def forward(self, x):
        B, C, H, W = x.shape
        embeddings, truth_patches, ids_unmasked, ids_masked = self.encode(x)
        ind_unmasked_dec = ids_unmasked.unsqueeze(-1).expand(-1, -1, self.D_decoder) # (B, n_keep, D)

        ## DECODER
        x = self.masked_embedding.repeat(B, self.n_patches, 1)
        # assert x.shape == (B, self.n_patches, self.D_decoder), x.shape
        unmasked_embeddings_dec = self.enc2dec_projection(embeddings)
        embeddings_dec = x.clone().scatter(1, ind_unmasked_dec, unmasked_embeddings_dec)
        # embeddings_dec = torch.scatter(x, 1, ind_unmasked_dec, unmasked_embeddings_dec)
        embeddings_dec = embeddings_dec + self.pos_embeddings_dec # broadcast along B in pos_embeddings
        x = self.decoder_norm1(embeddings_dec)
        x, _ = self.decoder_msa(x, x, x)
        embeddings_dec = embeddings_dec + x
        x = self.decoder_norm2(embeddings_dec)
        x = F.gelu( self.decoder_mlp1(x) )
        x = self.decoder_mlp2(x)
        embeddings_dec = embeddings_dec + x
        y_patches = self.dec2img_projection(embeddings_dec)

        return y_patches, truth_patches, ids_masked

# net = Net()
# net.to(device)