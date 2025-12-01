import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    """
    Transforms raw time series into Patches + Positional Encoding.
    Input: [Batch, Channels, Seq_Len]
    Output: [Batch * Channels, Num_Patches, d_model]
    """
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding
        
        # Project Patch_Len (16) -> d_model (128)
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        
        # Positional Embedding (Learnable)
        # Max limit 100 patches is enough for 336 seq_len / 16 patch_len
        self.position_embedding = nn.Parameter(torch.randn(1, 100, d_model))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Channels, Seq_Len]
        n_vars = x.shape[1]
        
        # 1. Padding (if sequence length isn't divisible by stride)
        if self.padding_patch == 'end':
            W = self.patch_len
            S = self.stride
            pad_len = (W - S)
            x = nn.functional.pad(x, (0, pad_len))
        
        # 2. Unfold (Patching)
        # Output: [Batch, Channels, Num_Patches, Patch_Len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # 3. Reshape for Channel Independence
        # We merge Batch and Channels dimensions
        # Output: [Batch * Channels, Num_Patches, Patch_Len]
        b, m, n_patches, p_len = x.shape
        x = x.reshape(b * m, n_patches, p_len)
        
        # 4. Project & Add Position
        x = self.value_embedding(x)
        x = x + self.position_embedding[:, :n_patches, :]
        
        return self.dropout(x), n_vars, n_patches

class SelfAttention(nn.Module):
    """
    Standard Single-Head Self-Attention
    """
    def __init__(self, d_model, d_head):
        super().__init__()
        self.d_head = d_head
        self.query = nn.Linear(d_model, d_head, bias=False)
        self.key = nn.Linear(d_model, d_head, bias=False)
        self.value = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Scaled Dot-Product Attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        attn_weights = torch.softmax(scores, dim=-1)
        
        output = attn_weights @ V
        return output

class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block with Residual Connections
    """
    def __init__(self, d_model):
        super().__init__()
        self.attention = SelfAttention(d_model, d_model) # Single Head for simplicity
        self.norm1 = nn.LayerNorm(d_model)
        
        # Expansion: 128 -> 512 -> 128
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Highway 1: Attention
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Highway 2: FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class PatchTST(nn.Module):
    """
    The Main Model: Patching -> Transformer Encoder -> Flatten Head
    """
    def __init__(self, seq_len, pred_len, patch_len=16, stride=8, d_model=128, n_layers=3):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 1. Patching
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, 'end', 0.1)
        
        # 2. Transformer Encoder (Stack of N blocks)
        self.encoder = nn.Sequential(
            *[EncoderBlock(d_model) for _ in range(n_layers)]
        )
        
        # 3. Flatten Head (The Predictor)
        # Calculates the number of patches to determine input size for the head
        # Logic: (Seq_Len - Patch_Len) / Stride + 1 (+ Padding correction)
        # We compute this dynamically in forward pass usually, but here we can infer.
        # For seq_len=336, patch=16, stride=8 -> ~42 patches
        
        # We define a 'head' that projects the flattened representation to the forecast
        # This will be initialized in forward pass lazily or calculated
        self.head = None # Lazy Init
        self.d_model = d_model

    def forward(self, x):
        # Input: [Batch, Seq_Len, Channels]
        
        # 1. Channel Independence Permutation
        # [Batch, Seq_Len, Channels] -> [Batch, Channels, Seq_Len]
        x = x.permute(0, 2, 1)
        
        # 2. Patching
        # Output: [Batch*Channels, Num_Patches, d_model]
        x, n_vars, n_patches = self.patch_embedding(x)
        
        # 3. Transformer Encoder
        x = self.encoder(x)
        
        # 4. Flatten Head
        # Output: [Batch*Channels, Num_Patches * d_model]
        x = x.reshape(x.shape[0], -1)
        
        # Lazy Initialization of the Head (Rigorous trick to avoid manual math)
        if self.head is None:
            flatten_dim = x.shape[1]
            self.head = nn.Linear(flatten_dim, self.pred_len).to(x.device)
            
        # 5. Forecast
        # Output: [Batch*Channels, Pred_Len]
        x = self.head(x)
        
        # 6. Reshape back to original format
        # [Batch*Channels, Pred_Len] -> [Batch, Channels, Pred_Len]
        x = x.reshape(-1, n_vars, self.pred_len)
        
        # [Batch, Channels, Pred_Len] -> [Batch, Pred_Len, Channels]
        return x.permute(0, 2, 1)

if __name__ == "__main__":
    # Rigorous Shape Check
    model = PatchTST(seq_len=336, pred_len=96)
    dummy_input = torch.rand(32, 336, 7) # Batch 32, History 336, 7 Variables
    output = model(dummy_input)
    
    print("Input Shape:", dummy_input.shape)
    print("Output Shape:", output.shape)
    
    expected_shape = (32, 96, 7)
    assert output.shape == expected_shape, "Shape Mismatch!"
    print("✅ PatchTST Architecture Verified.")