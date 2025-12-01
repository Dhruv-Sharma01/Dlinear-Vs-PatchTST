import torch
import torch.nn as nn
import math

class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    Normalizes the input window to mean 0, var 1.
    Denormalizes the output back to the original scale.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        # Learnable affine parameters (scaling and shifting)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        # x: [Batch, Seq_Len, Channels]
        # Calculate Mean/Std across the Time Dimension (dim=1)
        dim2reduce = 1
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class PatchEmbedding(nn.Module):
    # (Same as before - code condensed for brevity but functionally identical)
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = nn.Parameter(torch.randn(1, 100, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        if self.padding_patch == 'end':
            pad_len = (self.patch_len - self.stride)
            x = nn.functional.pad(x, (0, pad_len))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        b, m, n_patches, p_len = x.shape
        x = x.reshape(b * m, n_patches, p_len)
        x = self.value_embedding(x)
        x = x + self.position_embedding[:, :n_patches, :]
        return self.dropout(x), n_vars, n_patches

class SelfAttention(nn.Module):
    # (Same as before)
    def __init__(self, d_model, d_head):
        super().__init__()
        self.d_head = d_head
        self.query = nn.Linear(d_model, d_head, bias=False)
        self.key = nn.Linear(d_model, d_head, bias=False)
        self.value = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        attn_weights = torch.softmax(scores, dim=-1)
        output = attn_weights @ V
        return output

class EncoderBlock(nn.Module):
    # (Same as before)
    def __init__(self, d_model):
        super().__init__()
        self.attention = SelfAttention(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class PatchTST(nn.Module):
    """
    PatchTST with RevIN Integration
    """
    def __init__(self, seq_len, pred_len, patch_len=16, stride=8, d_model=128, n_layers=3, enc_in=7):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 1. RevIN (The Fix)
        self.revin = RevIN(enc_in)

        # 2. Patching
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, 'end', 0.1)
        
        # 3. Encoder
        self.encoder = nn.Sequential(
            *[EncoderBlock(d_model) for _ in range(n_layers)]
        )
        
        # 4. Flatten Head
        self.head = None 
        self.d_model = d_model

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        
        # --- STEP A: NORMALIZE (RevIN) ---
        x = self.revin(x, 'norm')

        # --- STEP B: Main Transformer Logic ---
        # 1. Permute for Channel Independence
        x = x.permute(0, 2, 1) # [Batch, Channels, Seq_Len]
        
        # 2. Patching
        x, n_vars, n_patches = self.patch_embedding(x)
        
        # 3. Encoder
        x = self.encoder(x)
        
        # 4. Head
        x = x.reshape(x.shape[0], -1)
        if self.head is None:
            flatten_dim = x.shape[1]
            self.head = nn.Linear(flatten_dim, self.pred_len).to(x.device)
        x = self.head(x)
        
        # 5. Reshape back
        x = x.reshape(-1, n_vars, self.pred_len)
        x = x.permute(0, 2, 1) # [Batch, Pred_Len, Channels]

        # --- STEP C: DENORMALIZE (RevIN) ---
        x = self.revin(x, 'denorm')
        
        return x

if __name__ == "__main__":
    # Rigorous Shape Check with RevIN
    model = PatchTST(seq_len=336, pred_len=96, enc_in=7)
    dummy_input = torch.rand(32, 336, 7)
    output = model(dummy_input)
    print("Input Shape:", dummy_input.shape)
    print("Output Shape:", output.shape)
    assert output.shape == (32, 96, 7)
    print("✅ PatchTST + RevIN Verified.")