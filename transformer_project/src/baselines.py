import torch
import torch.nn as nn

class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    Just one Linear layer. That's it.
    """
    def __init__(self, seq_len, pred_len, individual=False, enc_in=7):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompose the series
        self.decompsition = SeriesDecomp(25) # 25 is a standard kernel size for hourly data
        
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            # The core: Just mapping seq_len -> pred_len
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channel]
        
        # 1. Decompose
        seasonal_init, trend_init = self.decompsition(x)
        
        # 2. Permute for Linear Layer (Batch, Channel, Seq_Len)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # 3. Apply Linear Projection
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len], dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # 4. Add back together
        x = seasonal_output + trend_output
        
        return x.permute(0, 2, 1) # [Batch, Pred_Len, Channel]

if __name__ == '__main__':
    print("--- Testing DLinear Architecture ---")
    
    # 1. Simulate Data (Batch=32, Lookback=336, Channels=7)
    batch_size = 32
    seq_len = 336
    pred_len = 96
    enc_in = 7
    
    input_tensor = torch.rand(batch_size, seq_len, enc_in)
    # Generate random targets to simulate Ground Truth
    target_tensor = torch.rand(batch_size, pred_len, enc_in)
    
    print(f"Input Shape: {input_tensor.shape}")
    print(f"Target Shape: {target_tensor.shape}")

    # 2. Test Mode A: Shared Weights (Standard)
    print("\n[Test 1] Running DLinear (Shared Weights)...")
    model = DLinear(seq_len, pred_len, individual=False, enc_in=enc_in)
    output = model(input_tensor)
    print(f"Output Shape: {output.shape}")
    
    # Calculate Metrics (MSE/MAE)
    mse = torch.mean((output - target_tensor) ** 2)
    mae = torch.mean(torch.abs(output - target_tensor))
    print(f"Metrics (Random Init) -> MSE: {mse.item():.4f}, MAE: {mae.item():.4f}")
    
    # Rigorous Assertion
    expected_shape = (batch_size, pred_len, enc_in)
    assert output.shape == expected_shape, f"Shape Mismatch! Expected {expected_shape}, got {output.shape}"

    # 3. Test Mode B: Individual Weights (Per-Channel)
    print("\n[Test 2] Running DLinear (Individual Weights)...")
    model_ind = DLinear(seq_len, pred_len, individual=True, enc_in=enc_in)
    output_ind = model_ind(input_tensor)
    print(f"Output Shape: {output_ind.shape}")
    
    # Calculate Metrics
    mse_ind = torch.mean((output_ind - target_tensor) ** 2)
    mae_ind = torch.mean(torch.abs(output_ind - target_tensor))
    print(f"Metrics (Random Init) -> MSE: {mse_ind.item():.4f}, MAE: {mae_ind.item():.4f}")
    
    assert output_ind.shape == expected_shape, "Shape Mismatch in Individual Mode!"
    
    print("\n✅ DLinear passed all rigorous shape tests.")