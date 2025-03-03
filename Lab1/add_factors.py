# add_factors.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class Config:
    """
    Configuration class to centralize all hyperparameters and settings.
    """
    # File paths and names
    FOLDER_PATH = ""
    INPUT_FILE = "klines_BTC.csv"
    OUTPUT_FILE = "klines_BTC_with_factors.csv"
    VISUALIZATION_FILE = "btc_targets_analysis.png"
    
    # Technical indicator parameters
    WINDOW_SIZE = 75
    VOLATILITY_WINDOW = 24
    MA_WINDOWS = [7, 25, 99]
    EMA_SHORT = 12
    EMA_LONG = 26
    MACD_SIGNAL = 9
    RSI_PERIOD = 14
    BOLLINGER_WINDOW = 20
    BOLLINGER_STD = 2
    
    # Outlier detection parameters
    ZSCORE_THRESHOLD = 3
    ANOMALY_PERCENTILE = 95
    
    # Autoencoder parameters
    AUTOENCODER_LAYERS = [64, 32, 16, 32, 64]
    REGULARIZATION_FACTOR = 0.001
    DROPOUT_RATE = 0.2
    EPOCHS = 50
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.4
    EARLY_STOPPING_PATIENCE = 5
    
    # Random forest parameters
    RF_ESTIMATORS = 100
    RF_RANDOM_STATE = 42
    
    # Target classification thresholds
    PRICE_MOVEMENT_THRESHOLDS = [-2, -0.5, 0.5, 2]
    PRICE_MOVEMENT_LABELS = ['Significant Drop', 'Small Drop', 'Sideways', 'Small Rise', 'Significant Rise']
    
    # Train-test split parameters
    TEST_SIZE = 0.2
    FORECAST_HORIZON = 1


def compute_atr(df: pd.DataFrame, window_size: int) -> pd.Series:
    """
    Compute Average True Range (ATR) indicator.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLC price data
    window_size : int
        Rolling window size for ATR calculation
        
    Returns:
    --------
    pandas.Series: ATR values
    """
    high_low = df["high"] - df["low"]
    high_prev_close = abs(df["high"] - df["close"].shift(1))
    low_prev_close = abs(df["low"] - df["close"].shift(1))

    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    return true_range.rolling(window=window_size).mean()


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load, clean, and preprocess raw price data.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing raw price data
        
    Returns:
    --------
    pandas.DataFrame: Cleaned dataframe
    """
    # Load the data
    df_original = pd.read_csv(file_path)

    # Convert timestamp column to datetime
    df_original["open_time"] = pd.to_datetime(df_original["open_time"])
    df_original.sort_values(by="open_time", inplace=True)

    # Make a copy before modification
    df_cleaned = df_original.copy()

    # Handle Missing Values
    df_cleaned.dropna(inplace=True)

    # Ensure Correct Data Types
    numeric_cols = ["open", "high", "low", "close", "volume", 
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].astype(float)

    # Remove Duplicates
    df_cleaned.drop_duplicates(subset=["open_time"], keep="first", inplace=True)
    
    return df_cleaned


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add various technical indicators as features to the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLC price data
        
    Returns:
    --------
    pandas.DataFrame: DataFrame with added technical indicators
    """
    # Price-based features
    df['return'] = df['close'].pct_change()  # Price returns
    df['log_return'] = np.log(df['close']/df['close'].shift(1))  # Log returns
    df['volatility'] = df['log_return'].rolling(window=Config.VOLATILITY_WINDOW).std()  # Volatility
    df['price_range'] = (df['high'] - df['low']) / df['open']  # Normalized price range

    # Moving averages
    for window in Config.MA_WINDOWS:
        df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
    
    # ATR (Average True Range)
    df["atr"] = compute_atr(df, Config.WINDOW_SIZE)

    # MACD
    df['ema_12'] = df['close'].ewm(span=Config.EMA_SHORT).mean()
    df['ema_26'] = df['close'].ewm(span=Config.EMA_LONG).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=Config.MACD_SIGNAL).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=Config.RSI_PERIOD).mean()
    avg_loss = loss.rolling(window=Config.RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    bb_window = Config.BOLLINGER_WINDOW
    bb_std_dev = Config.BOLLINGER_STD
    df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
    bb_std = df['close'].rolling(window=bb_window).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * bb_std_dev)
    df['bb_lower'] = df['bb_middle'] - (bb_std * bb_std_dev)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volume features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
    df['volume_relative'] = df['volume'] / df['volume_ma_7']

    # OBV (On-Balance Volume)
    df['obv'] = 0
    df.loc[0, 'obv'] = df.loc[0, 'volume']
    for i in range(1, len(df)):
        if df.loc[i, 'close'] > df.loc[i-1, 'close']:
            df.loc[i, 'obv'] = df.loc[i-1, 'obv'] + df.loc[i, 'volume']
        elif df.loc[i, 'close'] < df.loc[i-1, 'close']:
            df.loc[i, 'obv'] = df.loc[i-1, 'obv'] - df.loc[i, 'volume']
        else:
            df.loc[i, 'obv'] = df.loc[i-1, 'obv']

    # Time-based features
    df['hour'] = df['open_time'].dt.hour
    df['day_of_week'] = df['open_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['month'] = df['open_time'].dt.month

    # Remove rows with NaN values that resulted from calculating indicators
    df.dropna(inplace=True)
    
    return df


def add_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add multiple target variables for both regression and classification tasks.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price data and technical indicators
        
    Returns:
    --------
    pandas.DataFrame: DataFrame with added target variables
    """
    # Target 1: Binary classification - will price go up in next period?
    df['target_binary'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Target 2: Regression - percentage price change in next period
    df['target_pct_change'] = df['close'].pct_change(periods=-1) * 100

    # Target 3: Regression - absolute price change in next period
    df['target_abs_change'] = df['close'].shift(-1) - df['close']

    # Target 4: Multi-class classification - categorize price movement
    thresholds = Config.PRICE_MOVEMENT_THRESHOLDS
    
    def categorize_movement(pct_change):
        for i, threshold in enumerate(thresholds):
            if pct_change < threshold:
                return i
        return len(thresholds)  # Last category

    df['target_multiclass'] = df['target_pct_change'].apply(categorize_movement)

    # Target 5: Regression - log return (often better for financial modeling)
    df['target_log_return'] = np.log(df['close'].shift(-1) / df['close'])

    # Target 6: Regression - normalized price change (compared to recent volatility)
    volatility = df['close'].rolling(window=Config.BOLLINGER_WINDOW).std()
    df['target_normalized_change'] = df['target_abs_change'] / volatility

    # Remove NaNs from target creation
    df.dropna(inplace=True)
    
    return df


def detect_outliers(df: pd.DataFrame) -> np.ndarray:
    """
    Detect outliers using statistical methods and autoencoder.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with features
        
    Returns:
    --------
    numpy.ndarray: Boolean mask where True indicates an outlier
    """
    # Get feature columns (exclude timestamps and targets)
    feature_cols = [col for col in df.columns if col not in [
        'open_time', 'hour', 'day_of_week', 'is_weekend', 'month', 
        'target_binary', 'target_pct_change', 'target_abs_change',
        'target_multiclass', 'target_log_return', 'target_normalized_change'
    ]]

    # Statistical outlier detection (Z-score method)
    def detect_outliers_zscore(df, columns, threshold=Config.ZSCORE_THRESHOLD):
        outliers_mask = np.zeros(len(df), dtype=bool)
        for col in columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            col_outliers = z_scores > threshold
            outliers_mask = outliers_mask | col_outliers
        return outliers_mask

    outliers_mask_zscore = detect_outliers_zscore(df, feature_cols)
    print(f"Z-score method identified {outliers_mask_zscore.sum()} outliers")

    # Autoencoder-based outlier detection
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])

    # Define autoencoder model with regularization
    input_dim = scaled_data.shape[1]
    layers = Config.AUTOENCODER_LAYERS
    
    # Build the autoencoder model
    autoencoder = keras.Sequential()
    # Encoder
    autoencoder.add(keras.layers.Dense(
        layers[0], activation='relu', input_shape=(input_dim,),
        kernel_regularizer=keras.regularizers.l2(Config.REGULARIZATION_FACTOR)))
    autoencoder.add(keras.layers.Dropout(Config.DROPOUT_RATE))
    
    for units in layers[1:len(layers)//2 + 1]:
        autoencoder.add(keras.layers.Dense(units, activation='relu'))
        
    # Decoder
    for units in layers[len(layers)//2 + 1:]:
        autoencoder.add(keras.layers.Dense(units, activation='relu'))
        
    autoencoder.add(keras.layers.Dense(input_dim, activation='linear'))

    # Compile model
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train the autoencoder with early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=Config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )

    autoencoder.fit(
        scaled_data, scaled_data,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        validation_split=Config.VALIDATION_SPLIT,
        callbacks=[early_stopping],
        verbose=1
    )

    # Compute reconstruction error
    reconstructed = autoencoder.predict(scaled_data)
    reconstruction_error = np.mean(np.abs(scaled_data - reconstructed), axis=1)

    # Define anomaly threshold
    threshold = np.percentile(reconstruction_error, Config.ANOMALY_PERCENTILE)
    
    # Combine outlier detection methods
    outliers_mask_combined = (reconstruction_error > threshold) | outliers_mask_zscore
    print(f"Combined methods identified {outliers_mask_combined.sum()} outliers")
    
    return outliers_mask_combined


def analyze_feature_importance(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Analyze feature importance for different target variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with features and targets
    feature_cols : list
        List of feature column names
        
    Returns:
    --------
    tuple: Dataframes with feature importance for binary and regression targets
    """
    X = df[feature_cols]
    
    # For binary classification
    y_binary = df['target_binary']
    clf = RandomForestClassifier(
        n_estimators=Config.RF_ESTIMATORS, 
        random_state=Config.RF_RANDOM_STATE
    )
    clf.fit(X, y_binary)
    
    binary_importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 important features for binary classification:")
    print(binary_importances.head(10))

    # For regression (percentage change)
    y_pct = df['target_pct_change']
    reg = RandomForestRegressor(
        n_estimators=Config.RF_ESTIMATORS, 
        random_state=Config.RF_RANDOM_STATE
    )
    reg.fit(X, y_pct)
    
    reg_importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': reg.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 important features for price change regression:")
    print(reg_importances.head(10))
    
    return binary_importances, reg_importances


def visualize_data_and_targets(df_original: pd.DataFrame, df_cleaned: pd.DataFrame, 
                              binary_importances: pd.DataFrame, reg_importances: pd.DataFrame):
    """
    Visualize various aspects of the dataset and targets.
    
    Parameters:
    -----------
    df_original : pandas.DataFrame
        Original unprocessed DataFrame
    df_cleaned : pandas.DataFrame
        Processed DataFrame with features and targets
    binary_importances : pandas.DataFrame
        Feature importances for binary classification
    reg_importances : pandas.DataFrame
        Feature importances for regression
    """
    plt.figure(figsize=(15, 15))

    # Plot 1: Original vs Cleaned Price
    plt.subplot(3, 2, 1)
    plt.plot(df_original["open_time"], df_original["close"], label="Original", alpha=0.5)
    plt.plot(df_cleaned["open_time"], df_cleaned["close"], label="Cleaned", alpha=0.8, color='red')
    plt.title("Original vs Cleaned Price")
    plt.legend()

    # Plot 2: Binary Target Distribution
    plt.subplot(3, 2, 2)
    df_cleaned['target_binary'].value_counts().plot(kind='bar')
    plt.title("Binary Target Distribution (Up/Down)")
    plt.xlabel("Price Direction (1=Up, 0=Down)")
    plt.ylabel("Count")

    # Plot 3: Percentage Change Distribution
    plt.subplot(3, 2, 3)
    plt.hist(df_cleaned['target_pct_change'], bins=50)
    plt.title("Price Percentage Change Distribution")
    plt.xlabel("Percentage Change (%)")
    plt.ylabel("Frequency")

    # Plot 4: Multi-class Target Distribution
    plt.subplot(3, 2, 4)
    df_cleaned['target_multiclass'].value_counts().sort_index().plot(kind='bar')
    plt.title("Multi-class Target Distribution")
    plt.xlabel("Price Movement Category")
    plt.ylabel("Count")
    plt.xticks(range(len(Config.PRICE_MOVEMENT_LABELS)), Config.PRICE_MOVEMENT_LABELS, rotation=45)

    # Plot 5: Correlation Between Different Target Variables
    plt.subplot(3, 2, 5)
    target_corr = df_cleaned[['target_binary', 'target_pct_change', 'target_abs_change', 
                             'target_multiclass', 'target_log_return', 'target_normalized_change']].corr()
    plt.imshow(target_corr, cmap='coolwarm')
    plt.colorbar()
    plt.title("Correlation Between Target Variables")
    plt.xticks(range(6), target_corr.columns, rotation=90)
    plt.yticks(range(6), target_corr.columns)

    # Plot 6: Top features importance (combined importance)
    plt.subplot(3, 2, 6)
    combined_importance = binary_importances.set_index('feature')['importance'] + reg_importances.set_index('feature')['importance']
    combined_importance.sort_values(ascending=False).head(10).plot(kind='bar')
    plt.title("Top Features (Combined Importance)")
    plt.tight_layout()
    plt.xlabel("Feature")
    plt.ylabel("Combined Importance Score")

    plt.tight_layout()
    plt.savefig(Config.VISUALIZATION_FILE)
    plt.show()


def prepare_train_test_splits(df: pd.DataFrame, test_size=None, forecast_horizon=None):
    """
    Prepare chronological train/test splits for time series data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The cleaned dataframe with features and targets
    test_size : float, optional
        Proportion of data to use for testing
    forecast_horizon : int, optional
        How many periods ahead to forecast
        
    Returns:
    --------
    dict containing train/test splits for different target variables
    """
    if test_size is None:
        test_size = Config.TEST_SIZE
        
    if forecast_horizon is None:
        forecast_horizon = Config.FORECAST_HORIZON
        
    # Feature columns (everything except targets and datetime)
    feature_cols = [col for col in df.columns if col not in [
        'open_time', 'target_binary', 'target_pct_change', 'target_abs_change',
        'target_multiclass', 'target_log_return', 'target_normalized_change'
    ]]
    
    # Split chronologically
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Training data from {train_df['open_time'].min()} to {train_df['open_time'].max()}")
    print(f"Testing data from {test_df['open_time'].min()} to {test_df['open_time'].max()}")
    
    # Prepare scalers
    feature_scaler = MinMaxScaler()
    X_train = feature_scaler.fit_transform(train_df[feature_cols])
    X_test = feature_scaler.transform(test_df[feature_cols])
    
    # Prepare regression target scalers
    regression_scalers = {}
    regression_targets = ['target_pct_change', 'target_abs_change', 
                          'target_log_return', 'target_normalized_change']
    
    y_train_dict = {}
    y_test_dict = {}
    
    # Classification targets
    y_train_dict['binary'] = train_df['target_binary'].values
    y_test_dict['binary'] = test_df['target_binary'].values
    
    y_train_dict['multiclass'] = train_df['target_multiclass'].values
    y_test_dict['multiclass'] = test_df['target_multiclass'].values
    
    # Regression targets (scaled)
    for target in regression_targets:
        scaler = MinMaxScaler()
        y_train_dict[target] = scaler.fit_transform(train_df[[target]])
        y_test_dict[target] = scaler.transform(test_df[[target]])
        regression_scalers[target] = scaler
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train_dict,
        'y_test': y_test_dict,
        'feature_scaler': feature_scaler,
        'regression_scalers': regression_scalers,
        'feature_columns': feature_cols,
        'train_dates': train_df['open_time'],
        'test_dates': test_df['open_time']
    }


def main():
    """
    Main function to execute the entire data processing pipeline.
    """
    # Step 1: Load and clean data
    df_cleaned = load_and_clean_data(Config.INPUT_FILE)
    df_original = df_cleaned.copy()  # Save original for visualization comparison
    
    # Step 2: Add technical indicators
    df_cleaned = add_technical_indicators(df_cleaned)
    
    # Step 3: Add target variables
    df_cleaned = add_target_variables(df_cleaned)
    
    # Step 4: Detect and remove outliers
    outliers_mask = detect_outliers(df_cleaned)
    df_cleaned = df_cleaned[~outliers_mask]
    
    # Get feature columns for analysis
    feature_cols = [col for col in df_cleaned.columns if col not in [
        'open_time', 'hour', 'day_of_week', 'is_weekend', 'month', 
        'target_binary', 'target_pct_change', 'target_abs_change',
        'target_multiclass', 'target_log_return', 'target_normalized_change'
    ]]
    
    # Step 5: Analyze feature importance
    binary_importances, reg_importances = analyze_feature_importance(df_cleaned, feature_cols)
    
    # Step 6: Visualize data and targets
    visualize_data_and_targets(df_original, df_cleaned, binary_importances, reg_importances)
    
    # Step 7: Save cleaned data with multiple targets
    df_cleaned.to_csv(Config.OUTPUT_FILE, index=False)
    
    # Step 8: Prepare train/test splits
    splits = prepare_train_test_splits(df_cleaned)
    print("\nSplits created for all target types.")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Original dataset shape: {df_original.shape}")
    print(f"Cleaned dataset shape with multiple targets: {df_cleaned.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Number of target variables: 6")


if __name__ == "__main__":
    main()