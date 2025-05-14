import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionBlock(nn.Module):
    """
    Self attention block with post-normalisation. Includes query, key, value masks.
    """
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, x_mask=None):

        # Multi-head self attention
        attn_output, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=x_mask
        )
        x = self.norm(x + attn_output)

        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, y, key_padding_mask=None):
        out, _ = self.attn(x, y, y, key_padding_mask=key_padding_mask)
        # out = out.last_hidden_state
        return self.norm(x + out)

class ImgCrossModalityEncoder(nn.Module):
    """
    Processes mask image features using text features to generate cross-attentions. It enhances
    the representation of images by interacting with the text modality. The image cross-modality
    encoder has 3 cross-attention blocks. It uses the outputs of the text encoder as keys and
    values to compute cross-attention.
    """
    def __init__(self, hidden_dim, num_heads, num_layers=3):
        super().__init__()
        self.self_attn = SelfAttentionBlock(hidden_dim, num_heads)
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

    def forward(self, vm, w, w_mask):
        # Self-attention
        vm = self.self_attn(vm)
        # Cross-attention
        # Text is key and value. We need to mask the padding tokens in the text
        for block in self.cross_blocks:
            vm = block(vm, w, w_mask)

        return self.ffn(vm)
    
class ImgCrossModalityDecoder(nn.Module):
    """
    Input: 
    - vm: masked image features after ImgCrossModalityEncoder as query
    - w: text features as key and value 
    - w_mask: text padding mask

    Output: [B, v_seq, D]
    - reconstructed image patches

    Layers:
    - 3 cross-attention blocks
    - 1 fully connected layer   
    """
    def __init__(self, hidden_dim, num_heads, num_layers=3, patch_size=16, channels=3):
        super().__init__()
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        # Predicts grayscale image patches no need for channels
        # self.patch_size = patch_size
        # self.channels = channels
        # create fully connected which predicts grayscale image patches
        self.fc = nn.Linear(hidden_dim, patch_size * patch_size * 1)

    def forward(self, vm, w, w_mask):
        # Cross-attention
        # Text is key and value. We need to mask the padding tokens in the text
        for block in self.cross_blocks:
            vm = block(vm, w, w_mask)

        vm = self.fc(vm)  # [B, v_seq, patch_size * patch_size]
        return vm

class TxtCrossModalityEncoder(nn.Module):
    """
    Text Cross-Modality Encoder (g_txt): Processes text features using image features to generate
    cross-attentions. It enhances the representation of text by interacting with the image
    modality. The text cross-modality encoder has 3 cross-attention blocks. It uses the outputs of
    the image encoder as keys and values to compute cross-attention. Uses post normalisation.
    """
    def __init__(self, hidden_dim, num_heads, num_layers=3):
        super().__init__()
        self.self_attn = SelfAttentionBlock(hidden_dim, num_heads)
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

    def forward(self, wm, v, w_mask):
        # Self-attention
        wm = self.self_attn(wm, w_mask)
        # Cross-attention
        # after text-cross attention, we only apply a classifier on the masked tokens
        # so no influence of the padding tokens -> no need for padding mask
        for block in self.cross_blocks:
            wm = block(wm, v, key_padding_mask=None)

        return self.ffn(wm)