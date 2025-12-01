import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding.
    Injects information about the relative or absolute position of the tokens in the sequence.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create a long enough PE matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Log-space calculation for stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as a buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        # Slice the PE matrix to the current sequence length
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention Mechanism.
    Splits d_model into n_heads to attend to different subspaces.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        batch_size, seq_len, _ = x.shape
        
        # 1. Project and Reshape to [Batch, Seq_Len, n_heads, d_head]
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        
        # 2. Transpose to [Batch, n_heads, Seq_Len, d_head] for Matrix Mult
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 3. Scaled Dot-Product Attention
        # Scores: [Batch, n_heads, Seq_Len, Seq_Len]
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply Dropout to attention weights
        attn_weights = self.dropout(attn_weights)
        
        # 4. Aggregate Values
        # Output: [Batch, n_heads, Seq_Len, d_head]
        context = attn_weights @ V
        
        # 5. Concatenate Heads
        # Transpose back: [Batch, Seq_Len, n_heads, d_head]
        context = context.transpose(1, 2)
        # Flatten: [Batch, Seq_Len, d_model]
        context = context.contiguous().view(batch_size, seq_len, self.d_model)
        
        # 6. Final Linear Projection
        output = self.out_proj(context)
        
        return output

class EncoderLayer(nn.Module):
    """
    Standard Transformer Encoder Layer: MHA -> Add & Norm -> FFN -> Add & Norm
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Sublayer 1: Multi-Head Attention
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Sublayer 2: Feed Forward Network
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x

class VanillaTransformer(nn.Module):
    """
    A Standard Transformer for Time Series Forecasting.
    Uses point-wise tokens (no patching) and Multi-Head Attention.
    """
    def __init__(self, enc_in, seq_len, pred_len, d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 1. Embedding
        # Projects raw features (e.g., 7) to d_model space
        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # 2. Encoder
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        
        # 3. Output Head (Direct Forecasting)
        # We flatten the entire sequence and project to the prediction horizon
        # Input: seq_len * d_model
        # Output: pred_len * c_out (same as enc_in for simplicity here)
        self.projection = nn.Linear(seq_len * d_model, pred_len * enc_in)
        self.enc_in = enc_in

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        batch_size, seq_len, channels = x.shape
        
        # 1. Embedding
        x = self.enc_embedding(x) # [Batch, Seq_Len, d_model]
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 2. Encoder Layers
        for layer in self.encoder:
            x = layer(x)
            
        # 3. Output Projection
        # Flatten: [Batch, Seq_Len * d_model]
        x = x.reshape(batch_size, -1)
        
        # Project: [Batch, Pred_Len * Channels]
        output = self.projection(x)
        
        # Reshape: [Batch, Pred_Len, Channels]
        output = output.reshape(batch_size, self.pred_len, self.enc_in)
        
        return output

if __name__ == "__main__":
    # Rigorous Shape Check
    # Batch=32, History=336, Channels=7, Forecast=96
    model = VanillaTransformer(enc_in=7, seq_len=336, pred_len=96, d_model=128, n_heads=4)
    dummy_input = torch.rand(32, 336, 7)
    
    output = model(dummy_input)
    
    print("Input Shape:", dummy_input.shape)
    print("Output Shape:", output.shape)
    
    expected_shape = (32, 96, 7)
    assert output.shape == expected_shape, "Shape Mismatch!"
    print("✅ Vanilla Multi-Head Transformer Verified.")