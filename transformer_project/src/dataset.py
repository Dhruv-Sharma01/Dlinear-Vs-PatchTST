import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class ETTDataset(Dataset):
    def __init__(self, csv_path, flag='train', seq_len=336, pred_len=96, target='OT'):
        """
        Args:
            csv_path: Path to ETTh1.csv
            flag: 'train', 'val', or 'test'
            seq_len: Lookback window (input)
            pred_len: Forecast horizon (label)
            target: The column we want to predict (OT)
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target = target
        self.flag = flag
        
        # 1. Load Data
        self.__read_data__(csv_path)

    def __read_data__(self, csv_path):
        df_raw = pd.read_csv(csv_path)
        
        # 2. Define Split Borders (Rigorous chronological split)
        # Train: 0-12 months | Val: 12-16 months | Test: 16-20 months
        # ETTh1 has ~17,420 hours total.
        type_map = {'train': 0, 'val': 1, 'test': 2}
        border1s = [0, 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        
        border1 = border1s[type_map[self.flag]]
        border2 = border2s[type_map[self.flag]]
        
        # 3. Standardization (Crucial for Transformers)
        # fit ONLY on train data to avoid Data Leakage
        cols_data = df_raw.columns[1:] # Exclude 'date'
        df_data = df_raw[cols_data]
        
        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler = StandardScaler()
        self.scaler.fit(train_data.values)
        
        # Transform the specific slice we need
        data = self.scaler.transform(df_data.values)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
    
    def __getitem__(self, index):
        # Sliding Window Logic
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1