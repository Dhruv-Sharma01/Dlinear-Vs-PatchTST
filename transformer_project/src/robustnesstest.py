import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# --- Imports matching your project structure ---
# We use 'src.' prefix assuming you run from the project root (python3 src/robustness_test.py)
from dataset import ETTDataset
from baselines import DLinear

# Prioritize loading the RevIN model architecture you created
try:
    from model_revinn import PatchTST
    print("Loaded PatchTST class from src.model_revinn")
except ImportError:
    from model import PatchTST
    print("Loaded PatchTST class from src.model")

def add_noise(x, noise_level):
    """Adds Gaussian noise to the input."""
    noise = torch.randn_like(x) * noise_level
    return x + noise

def mask_input(x, mask_ratio):
    """Randomly zeros out a percentage of time steps (Sensor Failure)."""
    mask = torch.rand_like(x) > mask_ratio
    return x * mask.float()

def evaluate_robustness(model, loader, device, attack_type='noise', levels=[0.1, 0.2]):
    model.eval()
    results = []
    
    print(f"--- Testing {attack_type.upper()} robustness ---")
    
    for level in levels:
        mse_list = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                
                # Apply Attack
                if attack_type == 'noise':
                    batch_x_corrupt = add_noise(batch_x, level)
                elif attack_type == 'mask':
                    batch_x_corrupt = mask_input(batch_x, level)
                
                # Inference
                outputs = model(batch_x_corrupt)
                
                # Metric
                mse = torch.mean((outputs - batch_y) ** 2).item()
                mse_list.append(mse)
        
        avg_mse = np.mean(mse_list)
        print(f"Level {level}: MSE = {avg_mse:.4f}")
        results.append(avg_mse)
        
    return results

if __name__ == "__main__":
    # Settings
    seq_len = 336
    pred_len = 96
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    print("Loading Test Data...")
    test_set = ETTDataset('./data/ETTh1.csv', flag='test', seq_len=seq_len, pred_len=pred_len)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, drop_last=False)

    # --- 1. Load DLinear ---
    print("\n[Loading DLinear]")
    model_dlinear = DLinear(seq_len=seq_len, pred_len=pred_len, enc_in=7).to(device)
    
    # Path based on your screenshot
    dlinear_path = 'best_model_baseline.pth'
    
    if os.path.exists(dlinear_path):
        model_dlinear.load_state_dict(torch.load(dlinear_path, map_location=device))
        print(f"✅ Loaded DLinear from {dlinear_path}")
    else:
        print(f"⚠️ Warning: '{dlinear_path}' not found. Using random weights!")

    # --- 2. Load PatchTST ---
    print("\n[Loading PatchTST]")
    # Initialize model (enc_in=7 required for RevIN)
    model_patch = PatchTST(seq_len=seq_len, pred_len=pred_len, enc_in=7, patch_len=16, stride=8, d_model=128).to(device)
    
    # CRITICAL FIX: Run dummy input to initialize lazy layers BEFORE loading weights
    print("Initializing lazy layers...")
    dummy_input = torch.rand(1, seq_len, 7).to(device)
    model_patch(dummy_input) 
    
    # Path based on your screenshot
    patch_path = 'best_model_patch_revin.pth'
    
    if os.path.exists(patch_path):
        model_patch.load_state_dict(torch.load(patch_path, map_location=device))
        print(f"✅ Loaded PatchTST from {patch_path}")
    else:
        print(f"⚠️ Warning: '{patch_path}' not found. Using random weights!")

    # --- 3. Run Experiments ---
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] # 0% to 50% noise/masking
    
    # Experiment A: Random Masking (Missing Data)
    print("\n=== Experiment A: Resilience to Missing Data (Masking) ===")
    dlinear_scores = evaluate_robustness(model_dlinear, test_loader, device, 'mask', noise_levels)
    patch_scores = evaluate_robustness(model_patch, test_loader, device, 'mask', noise_levels)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, dlinear_scores, marker='o', label='DLinear', color='blue')
    plt.plot(noise_levels, patch_scores, marker='s', label='PatchTST', color='orange')
    plt.title("Robustness to Missing Sensors (Random Masking)")
    plt.xlabel("Percentage of Data Missing (0.0 - 0.5)")
    plt.ylabel("Test MSE (Lower is Better)")
    plt.legend()
    plt.grid(True)
    plt.savefig('robustness_masking.png')
    print("\nGraph saved to 'robustness_masking.png'")