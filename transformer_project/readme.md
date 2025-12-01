Benchmarking Robustness: Linear vs. Transformer Architectures for Long-Horizon Industrial Forecasting

📄 Abstract

This research project conducts a rigorous comparative analysis between state-of-the-art Patch-Based Transformers (PatchTST) and efficient Linear Decomposition models (DLinear) on the ETTh1 industrial benchmark. The study evaluates performance across varying lookback windows (336h, 720h) and introduces an adversarial robustness evaluation to simulate sensor failure.

Key Findings:

Efficiency: On clean data, the simple DLinear model outperforms complex Transformers (MSE 0.3849 vs 0.3957), replicating the findings of Zeng et al. (2023).

Robustness: In simulated sensor failure scenarios (>10% data missing), PatchTST significantly outperforms DLinear, demonstrating that the self-attention mechanism provides critical resilience in hostile data environments.

Failure of Tradition: Classical statistical models (ARIMA) failed catastrophically (MSE 3.18) due to recursive error accumulation over long horizons.

📂 Repository Structure

The codebase is modularized for reproducibility, separating statistical baselines, linear architectures, and transformer logic.

TRANSFORMER_PROJECT/
├── data/
│   └── ETTh1.csv               # Dataset: Electricity Transformer Temperature (Hourly)
├── src/
│   ├── baselines.py            # Architecture: DLinear (Decomposition Linear)
│   ├── classicalmodels.py      # Architecture: ARIMA & GARCH (Statistical Control Group)
│   ├── dataset.py              # Data Pipeline: Sliding Window & Standardization
│   ├── model.py                # Architecture: Vanilla PatchTST
│   ├── model_revinn.py         # Architecture: PatchTST + RevIN (Reversible Instance Norm)
│   ├── vanilla_transformer.py  # Architecture: Standard Point-wise Transformer (Ablation)
│   ├── robustness_test.py      # Experiment: Adversarial Noise & Masking Evaluation
│   ├── train.py                # Training Loop for Transformers
│   ├── train_baseline.py       # Training Loop for Linear Models
│   └── testdataset.py          # Unit Tests for Data Pipeline
├── arima_garch_result.png      # Visualization: The Recursive Trap
├── robustness_masking.png      # Visualization: The Robustness Crossover
├── best_model_baseline.pth     # Checkpoint: Trained DLinear
├── best_model_patch_revin.pth  # Checkpoint: Trained PatchTST
└── requirements.txt            # Dependencies


🧠 Methodology & Architectures

1. The Challenge

Predicting Oil Temperature (OT) 96 hours into the future based on 336 hours of history using multivariate sensor data (Load, Voltage, etc.).

2. Models Evaluated

DLinear: A decomposition-based linear model that separates Trend and Seasonality. It uses fixed-weight matrices for direct multi-step forecasting.

PatchTST: A Transformer that segments time series into "Patches" (Tokens) to preserve local semantic meaning. Features Channel Independence and RevIN to handle distribution shift.

ARIMA: A classical autoregressive model used as a control group.

📊 Experimental Results

A. Clean Data Performance (Test Set)

Rank

Model Architecture

Test MSE

Test MAE

Training Time/Epoch

1

DLinear (Baseline)

0.3849

0.4033

~2.5s

2

PatchTST + RevIN

0.3957

0.4106

~107s

3

PatchTST (Vanilla)

0.4005

0.4194

~110s

4

ARIMA (Statistical)

3.1857

1.3925

N/A

Insight: On stable data, the Linear model is superior—4% more accurate and 45x faster to train.

B. Robustness Analysis (Adversarial)

We subjected both models to random sensor masking (0% to 50% missing data).

% Missing Data

DLinear MSE

PatchTST MSE

Winner

0%

0.3834

0.3936

Linear

10%

0.4063

0.3968

Transformer (Crossover)

30%

0.4843

0.4421

Transformer

50%

0.6048

0.5440

Transformer

Conclusion: DLinear is a "Glass Cannon"—excellent but fragile. PatchTST is resilient; its Attention mechanism allows it to infer missing contexts from neighboring patches.

🚀 How to Replicate

Setup Environment:

pip install -r requirements.txt


Run Statistical Baseline:

python3 src/classicalmodels.py


Generates arima_garch_result.png showing why ARIMA fails.

Train Models:

# Train the Linear Baseline
python3 src/train_baseline.py

# Train the Transformer
python3 src/train.py


Run Robustness Evaluation:

python3 src/robustness_test.py


Generates robustness_masking.png comparing model resilience.

📜 References

DLinear: Are Transformers Effective for Time Series Forecasting? (Zeng et al., 2023)

PatchTST: A Time Series is Worth 64 Words (Nie et al., 2023)

Dataset: ETT-Small (Zhou et al., 2021)