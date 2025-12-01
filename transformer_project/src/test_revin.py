import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# --- Project Imports ---
from dataset import ETTDataset
# Toggle this import to switch models!
from model_revinn import PatchTST 
# from src.vanilla_transformer import VanillaTransformer 

class EarlyStopping:
    """
    Stops training if validation loss doesn't improve after 'patience' epochs.
    Saves the best model checkpoint.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train():
    # --- Hyperparameters ---
    seq_len = 336
    pred_len = 96
    batch_size = 32
    learning_rate = 0.0001
    epochs = 20
    patience = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading Data...")
    train_set = ETTDataset('./data/ETTh1.csv', flag='train', seq_len=seq_len, pred_len=pred_len)
    val_set = ETTDataset('./data/ETTh1.csv', flag='val', seq_len=seq_len, pred_len=pred_len)
    test_set = ETTDataset('./data/ETTh1.csv', flag='test', seq_len=seq_len, pred_len=pred_len)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    # --- Model Initialization ---
    # NOTE: Ensure d_model, patch_len, etc. match your design
    model = PatchTST(
        seq_len=seq_len, 
        pred_len=pred_len, 
        patch_len=16, 
        stride=8, 
        d_model=128
    ).to(device)

    # --- Optimizer & Loss ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path='best_model_patch_revin.pth')
    
    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        train_loss = []
        epoch_time = time.time()
        
        for i, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            # Forward Pass
            # For PatchTST, we assume it outputs [Batch, Pred_Len, Channels]
            outputs = model(batch_x)

            # Loss Calculation
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f"\tIt: {i+1} | Loss: {loss.item():.7f}")

        # Validation
        model.eval()
        val_loss = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss.append(loss.item())

        train_loss = np.average(train_loss)
        val_loss = np.average(val_loss)

        print(f"Epoch: {epoch+1} | Cost: {time.time()-epoch_time:.2f}s | Train Loss: {train_loss:.7f} | Val Loss: {val_loss:.7f}")

        # Check Early Stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # --- Final Test ---
    print("\n--- Testing Best Model ---")
    model.load_state_dict(torch.load('best_model_patch_revin.pth'))
    model.eval()
    
    preds = []
    trues = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            outputs = model(batch_x)
            
            # Save predictions and truth for later visualization
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # Metrics
    mae = np.mean(np.abs(preds - trues))
    mse = np.mean((preds - trues) ** 2)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    # Remember our DLinear Baseline?
    # DLinear MSE was ~0.3 - 0.4 (if normalized correctly) or ~3.0 (if raw).
    # Since we used StandardScaler in dataset.py, these numbers are Normalized MSE.
    # A score < 0.4 is generally very good for this dataset.

if __name__ == "__main__":
    train()