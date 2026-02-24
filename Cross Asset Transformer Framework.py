import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
from Data_download import DataFetcherMT5, PrepareData
from datetime import datetime
import warnings
from sklearn import set_config
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import tensorflow as tf
from keras.layers import Input, MultiHeadAttention, LayerNormalization, Dense, Dropout, Concatenate, GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION AND INITIALIZATION
# =============================================================================

begin_date = datetime.strptime('2024.01.01', '%Y.%m.%d').date()
end_date = datetime.strptime('2024.08.01', '%Y.%m.%d').date()

set_config(transform_output="pandas")
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Trading symbols (multiple currency pairs)
SYMBOL = 'EURUSD'      # Primary symbol (target)
SYMBOL1 = 'XAUUSD'     # Gold
SYMBOL2 = 'GBPUSD'     # British Pound
SYMBOL3 = 'USDJPY'     # Japanese Yen
SYMBOL4 = 'USDCHF'     # Swiss Franc
SYMBOL5 = 'AUDUSD'     # Australian Dollar

TIMEFRAME = 'H1'       # Single timeframe for all symbols

# MT5 timeframe constants
M1 = mt5.TIMEFRAME_M1
M2 = mt5.TIMEFRAME_M2
M3 = mt5.TIMEFRAME_M3
M4 = mt5.TIMEFRAME_M4
M5 = mt5.TIMEFRAME_M5
M6 = mt5.TIMEFRAME_M6
M10 = mt5.TIMEFRAME_M10
M15 = mt5.TIMEFRAME_M15
M30 = mt5.TIMEFRAME_M30
H1 = mt5.TIMEFRAME_H1
H2 = mt5.TIMEFRAME_H2
H3 = mt5.TIMEFRAME_H3
H4 = mt5.TIMEFRAME_H4
H12 = mt5.TIMEFRAME_H12
D1 = mt5.TIMEFRAME_D1
W1 = mt5.TIMEFRAME_W1
MN1 = mt5.TIMEFRAME_MN1

# =============================================================================
# DATA FETCHING - MULTIPLE SYMBOLS
# =============================================================================

# Initialize MT5 connection
fetcher = DataFetcherMT5(login=5304234, password="MM8uRZO^1L", server="ICMarketsSC-MT5")

date_from = datetime(2025, 1, 15)
date_to = datetime(2025, 5, 27)

print("=" * 60)
print("FETCHING DATA FROM METATRADER 5")
print("=" * 60)
print(f"Primary symbol (target): {SYMBOL}")
print(f"Additional symbols: {SYMBOL1}, {SYMBOL2}, {SYMBOL3}, {SYMBOL4}, {SYMBOL5}")
print(f"Timeframe: {TIMEFRAME}")
print("-" * 60)

# Fetch H1 data for all symbols
print(f"Fetching {SYMBOL}...")
df_1_symbol = fetcher.fetch_data_candle(symbol=SYMBOL, timeframe=H1, number_of_candles=10000)

print(f"Fetching {SYMBOL2}...")
df_2_symbol = fetcher.fetch_data_candle(symbol=SYMBOL2, timeframe=H1, number_of_candles=10000)

print(f"Fetching {SYMBOL3}...")
df_3_symbol = fetcher.fetch_data_candle(symbol=SYMBOL3, timeframe=H1, number_of_candles=10000)

print(f"Fetching {SYMBOL4}...")
df_4_symbol = fetcher.fetch_data_candle(symbol=SYMBOL4, timeframe=H1, number_of_candles=10000)

print(f"Fetching {SYMBOL5}...")
df_6_symbol = fetcher.fetch_data_candle(symbol=SYMBOL5, timeframe=H1, number_of_candles=10000)

print("-" * 60)
print(f"Data fetched for {SYMBOL}: {len(df_1_symbol)} candles")
print(f"Data fetched for {SYMBOL2}: {len(df_2_symbol)} candles")
print(f"Data fetched for {SYMBOL3}: {len(df_3_symbol)} candles")
print(f"Data fetched for {SYMBOL4}: {len(df_4_symbol)} candles")
print(f"Data fetched for {SYMBOL5}: {len(df_6_symbol)} candles")

# Plot EURUSD close price (primary symbol)
plt.figure(figsize=(14, 6))
plt.plot(df_1_symbol['close'], label=f'{SYMBOL} Close', color='blue', alpha=0.7, linewidth=1.5)
plt.title(f'{SYMBOL} H1 Close Price (Primary Target Symbol)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Remove weekends from all dataframes
df_1_symbol = df_1_symbol[df_1_symbol.index.dayofweek < 5]
df_2_symbol = df_2_symbol[df_2_symbol.index.dayofweek < 5]
df_3_symbol = df_3_symbol[df_3_symbol.index.dayofweek < 5]
df_4_symbol = df_4_symbol[df_4_symbol.index.dayofweek < 5]
df_6_symbol = df_6_symbol[df_6_symbol.index.dayofweek < 5]

print("\nWeekends removed. New lengths:")
print(f"  {SYMBOL}: {len(df_1_symbol)}")
print(f"  {SYMBOL2}: {len(df_2_symbol)}")
print(f"  {SYMBOL3}: {len(df_3_symbol)}")
print(f"  {SYMBOL4}: {len(df_4_symbol)}")
print(f"  {SYMBOL5}: {len(df_6_symbol)}")

# =============================================================================
# FEATURE ENGINEERING - TECHNICAL INDICATORS FOR ALL SYMBOLS
# =============================================================================

feature = PrepareData()
print("\n" + "=" * 60)
print("GENERATING TECHNICAL INDICATORS FOR ALL SYMBOLS")
print("=" * 60)

# EURUSD - Primary symbol (base features without suffix)
print(f"\nProcessing {SYMBOL} (primary symbol, no suffix)...")
df_1_symbol = feature.calculate_market_regime_indicators(df_1_symbol)
df_1_symbol_1 = feature.feature_dataset_1(df_1_symbol).fillna(0)
df_1_symbol = pd.concat([df_1_symbol, df_1_symbol_1], axis=1).reset_index(drop=True)
df_1_symbol = df_1_symbol.loc[:, ~df_1_symbol.columns.duplicated()]
print(f"  {SYMBOL} features shape: {df_1_symbol.shape}")

# GBPUSD
print(f"\nProcessing {SYMBOL2}...")
df_2_symbol = feature.calculate_market_regime_indicators(df_2_symbol)
df_2_symbol_1 = feature.feature_dataset_1(df_2_symbol).fillna(0)
df_2_symbol = pd.concat([df_2_symbol, df_2_symbol_1], axis=1).add_suffix('_GBPUSD').reset_index(drop=True)
df_2_symbol = df_2_symbol.loc[:, ~df_2_symbol.columns.duplicated()]
print(f"  {SYMBOL2} features shape: {df_2_symbol.shape}")

# USDJPY
print(f"Processing {SYMBOL3}...")
df_3_symbol = feature.calculate_market_regime_indicators(df_3_symbol)
df_3_symbol_1 = feature.feature_dataset_1(df_3_symbol).fillna(0)
df_3_symbol = pd.concat([df_3_symbol, df_3_symbol_1], axis=1).add_suffix('_USDJPY').reset_index(drop=True)
df_3_symbol = df_3_symbol.loc[:, ~df_3_symbol.columns.duplicated()]
print(f"  {SYMBOL3} features shape: {df_3_symbol.shape}")

# USDCHF
print(f"Processing {SYMBOL4}...")
df_4_symbol = feature.calculate_market_regime_indicators(df_4_symbol)
df_4_symbol_1 = feature.feature_dataset_1(df_4_symbol).fillna(0)
df_4_symbol = pd.concat([df_4_symbol, df_4_symbol_1], axis=1).add_suffix('_USDCHF').reset_index(drop=True)
df_4_symbol = df_4_symbol.loc[:, ~df_4_symbol.columns.duplicated()]
print(f"  {SYMBOL4} features shape: {df_4_symbol.shape}")

# AUDUSD
print(f"Processing {SYMBOL5}...")
df_5_symbol = feature.calculate_market_regime_indicators(df_6_symbol)
df_5_symbol_1 = feature.feature_dataset_1(df_5_symbol).fillna(0)
df_5_symbol = pd.concat([df_5_symbol, df_5_symbol_1], axis=1).add_suffix('_AUDUSD').reset_index(drop=True)
df_5_symbol = df_5_symbol.loc[:, ~df_5_symbol.columns.duplicated()]
print(f"  {SYMBOL5} features shape: {df_5_symbol.shape}")

# =============================================================================
# TARGET CREATION (EURUSD only)
# =============================================================================

# Binary target: 1 if next price > current price, else 0 (for EURUSD only)
target = np.where(df_1_symbol['close'].shift(-1) > df_1_symbol['close'], 1, 0)

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)
unique, counts = np.unique(target[~np.isnan(target)], return_counts=True)
for val, count in zip(unique, counts):
    direction = "UP" if val == 1 else "DOWN"
    print(f"  {direction}: {count} ({count/len(target[~np.isnan(target)]):.1%})")
print(f"  Total samples: {len(target[~np.isnan(target)])}")

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

SEQ_LEN = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42
NUM_HEADS = 4
KEY_DIM = 64
FF_DIM = 128
DROPOUT_RATE = 0.5
EPOCHS = 100
BATCH_SIZE = 32

# =============================================================================
# DATA PREPROCESSING UTILITIES
# =============================================================================

def clean_data(df_list, target):
    """
    Remove inf/NaN values and synchronize indices across all dataframes.
    
    Parameters
    ----------
    df_list : list
        List of pandas DataFrames for different symbols
    target : array-like
        Target values for primary symbol
    
    Returns
    -------
    tuple
        Cleaned dataframes and synchronized target
    """
    # Convert infinity to NaN
    df_list = [df.replace([np.inf, -np.inf], np.nan) for df in df_list]
    
    # Remove rows with NaN in target
    if isinstance(target, (pd.DataFrame, pd.Series)):
        target = target.replace([np.inf, -np.inf], np.nan).dropna()
        df_list = [df.loc[target.index] for df in df_list]
    
    # Remove rows with NaN in data
    clean_dfs = []
    for df in df_list:
        df_clean = df.dropna(how='any')
        clean_dfs.append(df_clean)
    
    # Synchronize indices across all symbols
    common_index = clean_dfs[0].index
    for df in clean_dfs[1:]:
        common_index = common_index.intersection(df.index)
    
    clean_dfs = [df.loc[common_index] for df in clean_dfs]
    target = target.loc[common_index] if isinstance(target, (pd.DataFrame, pd.Series)) else target
    
    return clean_dfs, target

def prepare_data_with_preprocessing(df_list, target, n_features=10, n_components=5):
    """
    Full preprocessing pipeline with robust error handling.
    
    1. Data validation and cleaning
    2. Train/test split
    3. Scaling, feature selection, and PCA for each symbol
    
    Parameters
    ----------
    df_list : list
        List of DataFrames (one per symbol)
    target : array-like
        Target values
    n_features : int
        Number of features for SelectKBest
    n_components : int
        Number of PCA components
    
    Returns
    -------
    tuple
        Processed sequences and preprocessing objects
    """
    # Input validation
    if not df_list:
        raise ValueError("DataFrame list is empty")
    
    if len(df_list[0]) == 0:
        raise ValueError("First DataFrame is empty")
    
    if isinstance(target, (pd.DataFrame, pd.Series)) and len(target) == 0:
        raise ValueError("Target is empty")
    
    # Data cleaning
    df_list = [df.replace([np.inf, -np.inf], np.nan).fillna(0) for df in df_list]
    
    if isinstance(target, (pd.DataFrame, pd.Series)):
        target = target.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Index synchronization across all symbols
    common_index = df_list[0].index
    for df in df_list[1:]:
        common_index = common_index.intersection(df.index)
    
    if len(common_index) == 0:
        raise ValueError("No common indices between DataFrames")
    
    df_list = [df.loc[common_index] for df in df_list]
    if isinstance(target, (pd.DataFrame, pd.Series)):
        target = target.loc[common_index]
    
    # Safe train/test split
    if len(common_index) < 2:
        raise ValueError(f"Too little data ({len(common_index)} rows) for splitting")
    
    split_idx = max(1, int(len(common_index) * (1 - TEST_SIZE)))
    
    X_train_list = [df.iloc[:split_idx].copy() for df in df_list]
    X_test_list = [df.iloc[split_idx:].copy() for df in df_list]
    
    y_train = target.iloc[:split_idx] if isinstance(target, (pd.DataFrame, pd.Series)) else target[:split_idx]
    y_test = target.iloc[split_idx:] if isinstance(target, (pd.DataFrame, pd.Series)) else target[split_idx:]
    
    # Validate splits
    for data, name in zip([X_train_list, X_test_list], ['training', 'test']):
        if any(len(df) == 0 for df in data):
            raise ValueError(f"Empty {name} set after split")
    
    # Preprocessing pipeline for each symbol
    scalers, selectors, pca_models = [], [], []
    
    for i in range(len(X_train_list)):
        print(f"  Preprocessing symbol {i+1}/{len(X_train_list)}...")
        
        # Standard scaling
        scaler = StandardScaler()
        X_train_list[i] = scaler.fit_transform(X_train_list[i])
        X_test_list[i] = scaler.transform(X_test_list[i])
        scalers.append(scaler)
        
        # SelectKBest feature selection
        k = min(n_features, X_train_list[i].shape[1])
        selector = SelectKBest(f_classif, k=k)
        X_train_list[i] = selector.fit_transform(X_train_list[i], y_train)
        X_test_list[i] = selector.transform(X_test_list[i])
        selectors.append(selector)
        
        # PCA dimensionality reduction
        pca = PCA(n_components=min(n_components, X_train_list[i].shape[1]))
        X_train_list[i] = pca.fit_transform(X_train_list[i])
        X_test_list[i] = pca.transform(X_test_list[i])
        pca_models.append(pca)
    
    # Create sequences
    X_train_seq, y_train_seq = create_aligned_sequences(
        [pd.DataFrame(x) for x in X_train_list], 
        y_train
    )
    X_test_seq = create_aligned_sequences(
        [pd.DataFrame(x) for x in X_test_list]
    )
    
    return (X_train_seq, X_test_seq, y_train_seq, y_test, 
            scalers, selectors, pca_models)

def create_aligned_sequences(X_list, y=None):
    """
    Create aligned sequences from multiple data sources.
    
    Parameters
    ----------
    X_list : list
        List of DataFrames (one per symbol)
    y : array-like, optional
        Target values
    
    Returns
    -------
    tuple or list
        Sequences and optionally aligned labels
    """
    min_length = min(len(df) for df in X_list) - SEQ_LEN + 1
    sequences_list = []
    
    for df in X_list:
        seq = np.array([df.iloc[i:i+SEQ_LEN].values for i in range(min_length)])
        sequences_list.append(seq)
    
    if y is not None:
        if isinstance(y, np.ndarray):
            labels = y[SEQ_LEN-1 : SEQ_LEN-1 + min_length]
        else:
            labels = y.iloc[SEQ_LEN-1 : SEQ_LEN-1 + min_length]
        return sequences_list, labels
    return sequences_list

def build_transformer(input_shapes):
    """
    Build Transformer model for multi-symbol input.
    
    Each symbol has its own input branch with Multi-Head Attention,
    then all branches are concatenated for final classification.
    
    Parameters
    ----------
    input_shapes : list
        List of input shapes for each symbol
    
    Returns
    -------
    tf.keras.Model
        Compiled Transformer model
    """
    # Inputs for each symbol
    inputs = [Input(shape=shape) for shape in input_shapes]
    
    processed = []
    for inp in inputs:
        # Multi-Head Attention
        x = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM)(inp, inp)
        x = Dropout(DROPOUT_RATE)(x)
        x = LayerNormalization(epsilon=1e-6)(inp + x)  # Skip connection
        
        # Feed Forward Network
        ffn = Dense(FF_DIM, activation='tanh')(x)
        ffn = Dense(input_shapes[0][-1])(ffn)
        ffn = Dropout(DROPOUT_RATE)(ffn)
        x = LayerNormalization(epsilon=1e-6)(x + ffn)
        
        # Sequence reduction
        x = GlobalAveragePooling1D()(x)
        processed.append(x)
    
    # Concatenate all symbol branches
    merged = Concatenate()(processed) if len(processed) > 1 else processed[0]
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(merged)
    
    # Compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 60)
    print("MULTI-SYMBOL TRANSFORMER MODEL TRAINING")
    print("=" * 60)
    print(f"Primary symbol (target): {SYMBOL}")
    print(f"Supporting symbols: {len([SYMBOL2, SYMBOL3, SYMBOL4, SYMBOL5])} additional pairs")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Test size: {TEST_SIZE:.0%}")
    print("-" * 60)
    
    # Prepare data list (all symbols)
    df_list = [df_1_symbol, df_2_symbol, df_3_symbol, df_4_symbol, df_5_symbol]
    symbol_names = [SYMBOL, SYMBOL2, SYMBOL3, SYMBOL4, SYMBOL5]
    
    print("\nPreprocessing data...")
    (X_train_seq, X_test_seq, 
     y_train_seq, y_test, 
     scalers, selectors, pca_models) = prepare_data_with_preprocessing(
        df_list, 
        target,
        n_features=100,
        n_components=20
    )
    
    print("\n" + "-" * 60)
    print("DATA PREPROCESSING COMPLETE")
    print("-" * 60)
    print(f"Training sequences: {len(X_train_seq[0])}")
    print(f"Test sequences: {len(X_test_seq[0])}")
    print(f"Sequence shape per symbol: {X_train_seq[0].shape[1:]}")
    print(f"Number of symbols: {len(X_train_seq)}")
    
    # Build model
    print("\nBuilding Transformer model...")
    input_shapes = [seq.shape[1:] for seq in X_train_seq]
    model = build_transformer(input_shapes)
    model.summary()
    
    # Train model
    print("\n" + "=" * 60)
    print("TRAINING TRANSFORMER MODEL")
    print("=" * 60)
    
    history = model.fit(
        x=X_train_seq,
        y=y_train_seq,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_seq, y_test[SEQ_LEN-1:]),
        verbose=1
    )
    
    # =============================================================================
    # MODEL EVALUATION
    # =============================================================================
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Get predictions
    y_pred_proba = model.predict(X_test_seq).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    y_true = y_test[SEQ_LEN-1:]
    
    # Calculate metrics
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                 f1_score, roc_auc_score, confusion_matrix,
                                 classification_report)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nTest Accuracy:  {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1-Score:       {f1:.4f}")
    print(f"ROC-AUC:        {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              DOWN    UP")
    print(f"Actual DOWN   {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       UP      {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['DOWN', 'UP']))
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Multi-Symbol Transformer Training History', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Save model
    model.save('multi_symbol_transformer.h5')
    print("\nModel saved as 'multi_symbol_transformer.h5'")
    
    print("\n" + "=" * 60)
    print("MULTI-SYMBOL TRANSFORMER TRAINING COMPLETED")
    print("=" * 60)