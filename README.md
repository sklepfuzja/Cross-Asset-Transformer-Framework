# Multi-Symbol Transformer Framework for Financial Time Series Modeling

A modular, research-oriented Transformer framework for multi-asset financial time series prediction.

This repository provides a scalable deep learning architecture for cross-asset modeling, feature engineering experimentation, and flexible target construction across financial markets.

It is designed as a **framework**, not a fixed trading strategy.

---

# 🔬 Research Objective

Financial markets exhibit:

- Cross-asset dependencies  
- Non-stationarity  
- Regime shifts  
- High noise-to-signal ratios  

This project explores whether **multi-branch Transformer encoders** can capture cross-symbol temporal structure while maintaining modular preprocessing and flexible target definition.

The focus is on:

- Architecture modularity  
- Experimental flexibility  
- Reproducible modeling pipeline  
- Clean separation of data, features, and targets  

---

# 🏗️ System Architecture Overview

The framework follows a structured pipeline:

```
Multi-Asset Time Series Input
        ↓
Per-Symbol Feature Engineering
        ↓
Data Cleaning & Index Synchronization
        ↓
Train/Test Split
        ↓
Per-Symbol Preprocessing
  ├─ Scaling
  ├─ Feature Selection
  └─ Dimensionality Reduction
        ↓
Sequence Construction
        ↓
Multi-Branch Transformer Encoder
        ↓
Cross-Asset Feature Fusion
        ↓
Task-Specific Output Layer
```

---

# 🧠 Model Architecture – Multi-Branch Transformer

Each symbol (or asset) is processed independently through its own Transformer branch:

### Transformer Block
- MultiHeadAttention
- Residual connections
- Layer normalization
- Feed-forward network
- Dropout regularization
- GlobalAveragePooling

After independent encoding:

- Symbol embeddings are concatenated
- A dense output layer performs final prediction

The architecture supports:

- Binary classification  
- Multi-class classification  
- Regression  

---

# 📈 Feature Engineering

For each symbol:

- Market regime indicators  
- Custom technical feature set (`feature_dataset_1`)  
- Automatic feature deduplication  
- NaN / inf handling  
- Cross-symbol index alignment  

The feature layer is modular and can be extended with:

- Statistical features  
- Alternative alpha factors  
- Microstructure features  
- Order book signals  
- Alternative data sources  

---

# 🧮 Preprocessing Pipeline

Per-symbol independent pipeline:

1. **StandardScaler**
2. **SelectKBest (f_classif)**
3. **PCA**
4. Sequence construction (default length = 10)

This design allows:

- Replacing PCA with autoencoders  
- Disabling dimensionality reduction  
- Using custom feature selectors  
- Integrating scikit-learn pipelines  
- Plugging in learned feature extractors  

Each asset is transformed independently before cross-symbol fusion.

---

# 🎯 Target Design – Fully Customizable

The example implementation demonstrates binary direction prediction:

```
1 → Next candle close > Current close
0 → Otherwise
```

However, the framework is target-agnostic.

You can redefine the objective to support:

- Multi-horizon prediction (t+3, t+5, t+10)
- Return threshold classification
- Volatility breakout detection
- Regression (next return value)
- Multi-class direction modeling
- Strategy-specific labeling logic
- Risk-adjusted targets

Target construction is explicitly defined in the main script and can be modified in a few lines.

---

# 🌍 Data & Instruments

Although the example uses Forex H1 data, the framework supports:

- Forex
- Crypto
- Equities
- Commodities
- Indices
- Cross-market modeling
- Any aligned numerical time series dataset

Data can come from:

- MetaTrader 5
- CSV files
- Exchange APIs
- Institutional data feeds
- Custom research datasets

As long as aligned feature matrices are provided, the architecture remains unchanged.

---

# ⚙️ Default Hyperparameters (Example Configuration)

| Parameter | Value |
|------------|--------|
| Sequence Length | 10 |
| Test Split | 20% |
| Epochs | 100 |
| Batch Size | 32 |
| Attention Heads | 4 |
| Key Dimension | 64 |
| Feed-Forward Dimension | 128 |
| Dropout | 0.5 |

All parameters are configurable.

---

# 📊 Evaluation

The framework supports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

It can be extended with:

- Walk-forward validation
- Time-series cross-validation
- Transaction cost modeling
- Risk-adjusted metrics
- Portfolio-level evaluation

---

# 📁 Project Structure

```
.
├── Cross Asset Transformer Framework.py          # Main multi-symbol Transformer training script
├── Data_download.py            # Data fetching & feature utilities
├── multi_symbol_transformer.h5 # Saved model (example output)
└── README.md
```

---

# 🚀 How to Run

```
python Cross Asset Transformer Framework.py
```

Requirements:

- Python 3.9+
- TensorFlow / Keras
- scikit-learn
- pandas
- numpy
- matplotlib

---

# 📄 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this framework, provided that the original license is included.

---

# ⚠️ Disclaimer

This repository is provided for research and educational purposes only.

It does not constitute:

- Financial advice  
- Investment advice  
- Trading recommendations  

Financial markets involve substantial risk.

The author assumes no responsibility for:

- Financial losses  
- Trading decisions  
- Production deployment without proper validation  

Any model derived from this framework should be rigorously tested using:

- Out-of-sample validation  
- Walk-forward analysis  
- Risk management constraints  
- Transaction cost assumptions  

Use responsibly.
