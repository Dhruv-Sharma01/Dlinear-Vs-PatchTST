from dataset import ETTDataset
from torch.utils.data import DataLoader

# Create Dataset
train_dataset = ETTDataset(csv_path='./data/ETTh1.csv', flag='train', seq_len=336, pred_len=96)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Fetch one batch
batch_x, batch_y = next(iter(train_loader))

print("Batch X Shape:", batch_x.shape) 
# Should be [32, 336, 7] (Batch, Lookback, Features)
print("Batch Y Shape:", batch_y.shape) 
# Should be [32, 96, 7]  (Batch, Horizon, Features)