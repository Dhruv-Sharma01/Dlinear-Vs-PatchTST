import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# Suppress warnings for cleaner output during iterative fitting
warnings.filterwarnings("ignore")

def load_single_series(csv_path, target_col='OT'):
    """
    Loads just the target column for univariate analysis.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    
    # We focus only on the target (Oil Temperature) for ARIMA
    series = df[target_col].values
    return series

def fit_arima(train_data, order=(5, 1, 0), forecast_steps=96):
    """
    Fits ARIMA and forecasts 'steps' ahead recursively.
    Order (p,d,q) is hardcoded here for speed, but normally found via AutoARIMA.
    """
    # 1. Fit Model
    # disp=0 turns off convergence output
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    
    # 2. Forecast
    forecast_result = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_result.predicted_mean
    
    return forecast_mean, model_fit.resid

def fit_garch(residuals):
    """
    Fits GARCH(1,1) on the residuals of the ARIMA model to model volatility.
    """
    # Rescale residuals to avoid convergence issues if they are too small
    scaling_factor = 100
    rescaled_resid = residuals * scaling_factor
    
    # Volatility model (GARCH)
    garch = arch_model(rescaled_resid, vol='Garch', p=1, q=1)
    garch_fit = garch.fit(disp='off')
    
    # Forecast Volatility (Variance)
    # horizon=96 matches our ARIMA forecast
    variance_forecast = garch_fit.forecast(horizon=96)
    
    # We take the variance of the LAST step to project forward
    # Note: GARCH forecasting is tricky; usually we want the variance 
    # for the same horizon as the mean forecast.
    return variance_forecast.variance.values[-1, :] / (scaling_factor**2)

if __name__ == "__main__":
    # --- Configuration ---
    csv_path = './data/ETTh1.csv'
    seq_len = 336  # Context Window
    pred_len = 96  # Prediction Horizon
    
    # 1. Load Data
    full_series = load_single_series(csv_path)
    
    # 2. Create a "Test Slice"
    # We will simulate being at a specific point in time (e.g., 80% through the dataset)
    test_index = int(len(full_series) * 0.8)
    
    train_slice = full_series[test_index - seq_len : test_index] # The "History" (336 hours)
    ground_truth = full_series[test_index : test_index + pred_len] # The "Future" (96 hours)
    
    print(f"\n--- Running Classical Benchmark ---")
    print(f"History Length: {len(train_slice)}")
    print(f"Forecast Horizon: {len(ground_truth)}")
    
    # 3. Run ARIMA (Mean Prediction)
    print("\n[1] Fitting ARIMA(5,1,0)... (This is recursive, might be slow)")
    pred_mean, residuals = fit_arima(train_slice, order=(5,1,0), forecast_steps=pred_len)
    
    # 4. Run GARCH (Volatility Prediction)
    print("[2] Fitting GARCH(1,1) on residuals...")
    pred_variance = fit_garch(residuals)
    pred_std = np.sqrt(pred_variance) # Convert variance to std dev
    
    # 5. Evaluation
    mse = mean_squared_error(ground_truth, pred_mean)
    mae = mean_absolute_error(ground_truth, pred_mean)
    
    print(f"\n--- Results ---")
    print(f"ARIMA MSE: {mse:.4f}")
    print(f"ARIMA MAE: {mae:.4f}")
    
    # 6. Visualization
    # We plot the history, the true future, and the ARIMA prediction with GARCH confidence intervals
    plt.figure(figsize=(12, 6))
    
    # Plot History
    plt.plot(np.arange(-seq_len, 0), train_slice, label='History', color='gray', alpha=0.5)
    
    # Plot Ground Truth
    plt.plot(np.arange(0, pred_len), ground_truth, label='Ground Truth', color='green')
    
    # Plot ARIMA Prediction
    plt.plot(np.arange(0, pred_len), pred_mean, label='ARIMA Forecast', color='red', linestyle='--')
    
    # Plot Confidence Intervals (2 Sigma ~ 95%) derived from GARCH volatility
    plt.fill_between(
        np.arange(0, pred_len),
        pred_mean - 2 * pred_std,
        pred_mean + 2 * pred_std,
        color='red', alpha=0.1, label='GARCH 95% CI'
    )
    
    plt.title(f"ARIMA + GARCH Forecast (Horizon={pred_len})")
    plt.xlabel("Hours (Relative to Now)")
    plt.ylabel("Oil Temperature")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save graph for report
    plt.savefig('arima_garch_result.png')
    print("\nGraph saved to 'arima_garch_result.png'. Check it to see the 'Recursive Trap'.")