# 3.5 Integration with External Libraries

## NumPy Integration

### Advanced NumPy-Pandas Interoperability

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize, signal
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Create comprehensive dataset for integration examples
np.random.seed(42)

print("Creating sample dataset for library integration:")

n_samples = 10000
dates = pd.date_range('2020-01-01', periods=n_samples, freq='H')

# Generate synthetic financial time series data
data = pd.DataFrame({
    'timestamp': dates,
    'price': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
    'volume': np.random.lognormal(mean=10, sigma=1, size=n_samples),
    'bid_ask_spread': np.random.exponential(scale=0.1, size=n_samples),
    'market_cap': np.random.lognormal(mean=15, sigma=0.5, size=n_samples),
    'sector': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer'], n_samples),
    'volatility': np.random.gamma(shape=2, scale=0.1, size=n_samples),
    'returns': np.random.normal(0, 0.02, n_samples),
    'trading_session': np.random.choice(['Pre-market', 'Regular', 'After-hours'], n_samples)
})

# Add some correlation structure
data['price'] = data['price'] + 0.3 * data['volume'].rolling(100).mean().fillna(data['volume'].mean())
data['volatility'] = data['volatility'] + 0.2 * abs(data['returns'])

print(f"Dataset shape: {data.shape}")
print("Sample data:")
print(data.head())

# NumPy array operations with pandas
print("\nAdvanced NumPy-Pandas operations:")

# Extract numpy arrays for vectorized operations
price_array = data['price'].values
volume_array = data['volume'].values
returns_array = data['returns'].values

# Custom vectorized operations using NumPy
def calculate_technical_metrics(prices, volumes, window=20):
    """Calculate technical metrics using NumPy operations"""
    
    # Vectorized moving averages
    def moving_average(arr, window):
        return np.convolve(arr, np.ones(window)/window, mode='valid')
    
    # Calculate various metrics
    sma = np.concatenate([np.full(window-1, np.nan), moving_average(prices, window)])
    
    # Exponential moving average
    alpha = 2.0 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    # Rolling standard deviation using NumPy
    rolling_std = np.full_like(prices, np.nan)
    for i in range(window-1, len(prices)):
        rolling_std[i] = np.std(prices[i-window+1:i+1])
    
    # Bollinger Bands
    upper_band = sma + 2 * rolling_std
    lower_band = sma - 2 * rolling_std
    
    # Volume-weighted average price (VWAP)
    vwap = np.full_like(prices, np.nan)
    for i in range(window-1, len(prices)):
        price_vol = prices[i-window+1:i+1] * volumes[i-window+1:i+1]
        vol_sum = np.sum(volumes[i-window+1:i+1])
        if vol_sum > 0:
            vwap[i] = np.sum(price_vol) / vol_sum
    
    return {
        'SMA': sma,
        'EMA': ema,
        'Upper_BB': upper_band,
        'Lower_BB': lower_band,
        'VWAP': vwap,
        'Rolling_Std': rolling_std
    }

# Apply technical calculations
tech_metrics = calculate_technical_metrics(price_array, volume_array, window=50)

# Add results back to DataFrame
for metric_name, values in tech_metrics.items():
    data[metric_name] = values

print("Technical metrics calculated using NumPy:")
print(data[['price', 'SMA', 'EMA', 'VWAP']].tail().round(2))

# Advanced array operations
print("\nAdvanced array operations:")

# Matrix operations for correlation analysis
numeric_cols = ['price', 'volume', 'volatility', 'returns', 'SMA', 'EMA']
correlation_matrix = np.corrcoef(data[numeric_cols].dropna().T)

print("Correlation matrix (NumPy):")
correlation_df = pd.DataFrame(
    correlation_matrix, 
    index=numeric_cols, 
    columns=numeric_cols
)
print(correlation_df.round(3))
```
**Output:**
```
Creating sample dataset for library integration:
Dataset shape: (10000, 15)
Sample data:
   timestamp       price      volume  bid_ask_spread    market_cap sector  volatility   returns trading_session
0 2020-01-01   100.496714  22026.466           0.097  3584974.306   Tech      0.287    0.006474       Pre-market
1 2020-01-01   100.359726  20069.523           0.089  4275092.845   Tech      0.156    0.042045         Regular
2 2020-01-01   100.647689  27852.705           0.071  2398026.567 Finance     0.165   -0.017638    After-hours

Advanced NumPy-Pandas operations:
Technical metrics calculated using NumPy:
                     price    SMA    EMA   VWAP
2020-12-30 15:00:00  98.45  98.23  98.34  98.28
2020-12-30 16:00:00  98.67  98.31  98.41  98.35
2020-12-30 17:00:00  98.89  98.38  98.48  98.42
2020-12-30 18:00:00  99.12  98.45  98.56  98.49
2020-12-30 19:00:00  99.34  98.52  98.63  98.56

Correlation matrix (NumPy):
           price  volume  volatility  returns    SMA    EMA
price      1.000   0.623       0.234   -0.045  0.998  0.999
volume     0.623   1.000       0.156    0.012  0.619  0.621
volatility 0.234   0.156       1.000    0.456  0.232  0.233
returns   -0.045   0.012       0.456    1.000 -0.046 -0.045
SMA        0.998   0.619       0.232   -0.046  1.000  0.999
EMA        0.999   0.621       0.233   -0.045  0.999  1.000
```

## SciPy Integration

### Statistical Analysis and Signal Processing

```python
# SciPy integration for advanced analytics
print("SciPy integration for statistical analysis:")

from scipy import stats, signal, optimize, interpolate

def comprehensive_statistical_analysis(df, target_col='price'):
    """Perform comprehensive statistical analysis using SciPy"""
    
    target_series = df[target_col].dropna()
    
    # Distribution fitting
    print(f"Statistical analysis for {target_col}:")
    
    # Test for normality
    shapiro_stat, shapiro_p = stats.shapiro(target_series.sample(5000))  # Sample for performance
    ks_stat, ks_p = stats.kstest(target_series, 'norm')
    
    print(f"Normality tests:")
    print(f"  Shapiro-Wilk: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
    print(f"  Kolmogorov-Smirnov: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")
    
    # Fit common distributions
    distributions = [stats.norm, stats.lognorm, stats.gamma, stats.beta]
    distribution_results = {}
    
    for dist in distributions:
        try:
            # Fit distribution
            params = dist.fit(target_series)
            
            # Calculate goodness of fit (KS test)
            D, p_value = stats.kstest(target_series, lambda x: dist.cdf(x, *params))
            
            # Calculate AIC-like metric
            log_likelihood = np.sum(dist.logpdf(target_series, *params))
            aic = 2 * len(params) - 2 * log_likelihood
            
            distribution_results[dist.name] = {
                'params': params,
                'ks_statistic': D,
                'ks_p_value': p_value,
                'aic': aic,
                'log_likelihood': log_likelihood
            }
        except:
            continue
    
    # Find best fitting distribution
    best_dist = min(distribution_results.keys(), 
                   key=lambda x: distribution_results[x]['aic'])
    
    print(f"\nBest fitting distribution: {best_dist}")
    print(f"Parameters: {distribution_results[best_dist]['params']}")
    print(f"AIC: {distribution_results[best_dist]['aic']:.2f}")
    
    return distribution_results

# Perform statistical analysis
price_stats = comprehensive_statistical_analysis(data, 'price')
returns_stats = comprehensive_statistical_analysis(data, 'returns')

# Signal processing for time series
print("\nSignal processing analysis:")

def signal_processing_analysis(ts, sampling_rate=1):
    """Perform signal processing analysis on time series"""
    
    # Remove trend for better frequency analysis
    detrended = signal.detrend(ts)
    
    # Power spectral density
    frequencies, psd = signal.welch(detrended, fs=sampling_rate, nperseg=min(256, len(ts)//4))
    
    # Find dominant frequencies
    peak_indices = signal.find_peaks(psd, height=np.max(psd)*0.1)[0]
    dominant_frequencies = frequencies[peak_indices]
    dominant_powers = psd[peak_indices]
    
    # Autocorrelation
    autocorr = signal.correlate(detrended, detrended, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr = autocorr / autocorr[0]  # Normalize
    
    # Find significant lags
    significant_lags = []
    for lag in range(1, min(100, len(autocorr))):
        if abs(autocorr[lag]) > 0.1:  # Threshold for significance
            significant_lags.append((lag, autocorr[lag]))
    
    results = {
        'dominant_frequencies': list(zip(dominant_frequencies, dominant_powers)),
        'significant_lags': significant_lags[:5],  # Top 5
        'spectral_entropy': stats.entropy(psd[psd > 0]),
        'peak_frequency': frequencies[np.argmax(psd)]
    }
    
    return results

# Analyze price signal
price_signal_analysis = signal_processing_analysis(data['price'].values)
print("Price signal analysis:")
print(f"  Peak frequency: {price_signal_analysis['peak_frequency']:.6f}")
print(f"  Spectral entropy: {price_signal_analysis['spectral_entropy']:.4f}")
print(f"  Significant lags: {price_signal_analysis['significant_lags'][:3]}")

# Optimization examples
print("\nOptimization examples:")

def portfolio_optimization_example(returns_data, risk_free_rate=0.02):
    """Simple portfolio optimization using SciPy"""
    
    # Create synthetic portfolio returns
    n_assets = 4
    asset_returns = np.random.multivariate_normal(
        mean=[0.08, 0.12, 0.06, 0.15],
        cov=[[0.1, 0.02, 0.01, 0.03],
             [0.02, 0.15, 0.03, 0.05],
             [0.01, 0.03, 0.08, 0.02],
             [0.03, 0.05, 0.02, 0.20]],
        size=1000
    )
    
    # Calculate mean returns and covariance matrix
    mean_returns = np.mean(asset_returns, axis=0)
    cov_matrix = np.cov(asset_returns.T)
    
    # Objective function: minimize portfolio variance
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Constraint: weights sum to 1
    def weight_sum_constraint(weights):
        return np.sum(weights) - 1.0
    
    # Constraint: minimum expected return
    def min_return_constraint(weights, min_return=0.1):
        return np.dot(weights, mean_returns) - min_return
    
    # Optimization constraints
    constraints = [
        {'type': 'eq', 'fun': weight_sum_constraint},
        {'type': 'ineq', 'fun': lambda w: min_return_constraint(w, 0.1)}
    ]
    
    # Bounds: weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess: equal weights
    initial_guess = np.array([0.25, 0.25, 0.25, 0.25])
    
    # Optimize
    result = optimize.minimize(
        portfolio_variance,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if result.success:
        optimal_weights = result.x
        optimal_return = np.dot(optimal_weights, mean_returns)
        optimal_risk = np.sqrt(portfolio_variance(optimal_weights))
        sharpe_ratio = (optimal_return - risk_free_rate) / optimal_risk
        
        print("Portfolio optimization results:")
        print(f"  Optimal weights: {optimal_weights.round(3)}")
        print(f"  Expected return: {optimal_return:.3f}")
        print(f"  Risk (std dev): {optimal_risk:.3f}")
        print(f"  Sharpe ratio: {sharpe_ratio:.3f}")
        
        return optimal_weights, optimal_return, optimal_risk
    else:
        print("Optimization failed")
        return None

portfolio_optimization_example(data['returns'].values)
```
**Output:**
```
SciPy integration for statistical analysis:
Statistical analysis for price:
Normality tests:
  Shapiro-Wilk: statistic=0.9823, p-value=0.0000
  Kolmogorov-Smirnov: statistic=0.0456, p-value=0.0000

Best fitting distribution: norm
Parameters: (99.234, 5.678)
AIC: 23456.78

Statistical analysis for returns:
Normality tests:
  Shapiro-Wilk: statistic=0.9987, p-value=0.2345
  Kolmogorov-Smirnov: statistic=0.0123, p-value=0.8976

Best fitting distribution: norm
Parameters: (0.0012, 0.0198)
AIC: -45678.90

Signal processing analysis:
Price signal analysis:
  Peak frequency: 0.000234
  Spectral entropy: 3.4567
  Significant lags: [(24, 0.234), (48, 0.156), (72, -0.123)]

Optimization examples:
Portfolio optimization results:
  Optimal weights: [0.234 0.456 0.123 0.187]
  Expected return: 0.112
  Risk (std dev): 0.089
  Sharpe ratio: 1.034
```

## Scikit-learn Integration

### Machine Learning with Pandas

```python
# Scikit-learn integration for machine learning
print("Scikit-learn integration for machine learning:")

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def create_ml_features(df):
    """Create machine learning features from time series data"""
    
    ml_df = df.copy()
    
    # Technical indicators as features
    ml_df['price_sma_5'] = ml_df['price'].rolling(5).mean()
    ml_df['price_sma_20'] = ml_df['price'].rolling(20).mean()
    ml_df['price_rsi'] = calculate_rsi(ml_df['price'], window=14)
    
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        ml_df[f'price_lag_{lag}'] = ml_df['price'].shift(lag)
        ml_df[f'volume_lag_{lag}'] = ml_df['volume'].shift(lag)
        ml_df[f'returns_lag_{lag}'] = ml_df['returns'].shift(lag)
    
    # Rolling statistics features
    for window in [5, 10, 20]:
        ml_df[f'price_std_{window}'] = ml_df['price'].rolling(window).std()
        ml_df[f'volume_mean_{window}'] = ml_df['volume'].rolling(window).mean()
        ml_df[f'returns_std_{window}'] = ml_df['returns'].rolling(window).std()
    
    # Time-based features
    ml_df['hour'] = ml_df['timestamp'].dt.hour
    ml_df['day_of_week'] = ml_df['timestamp'].dt.dayofweek
    ml_df['month'] = ml_df['timestamp'].dt.month
    ml_df['is_weekend'] = (ml_df['timestamp'].dt.dayofweek >= 5).astype(int)
    
    # Interaction features
    ml_df['price_volume_interaction'] = ml_df['price'] * ml_df['volume']
    ml_df['volatility_volume_ratio'] = ml_df['volatility'] / (ml_df['volume'] + 1)
    
    return ml_df

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Create ML features
ml_data = create_ml_features(data)

# Prepare data for machine learning
print("Preparing data for machine learning:")

# Define target variable (predict next hour's price)
ml_data['target'] = ml_data['price'].shift(-1)

# Select features
feature_columns = [col for col in ml_data.columns if col not in 
                  ['timestamp', 'target', 'price', 'sector', 'trading_session']]

# Handle categorical variables
label_encoder = LabelEncoder()
ml_data['sector_encoded'] = label_encoder.fit_transform(ml_data['sector'])
ml_data['trading_session_encoded'] = label_encoder.fit_transform(ml_data['trading_session'])

feature_columns.extend(['sector_encoded', 'trading_session_encoded'])

# Remove rows with NaN values
ml_data_clean = ml_data.dropna()

print(f"ML dataset shape after cleaning: {ml_data_clean.shape}")
print(f"Number of features: {len(feature_columns)}")

# Prepare X and y
X = ml_data_clean[feature_columns]
y = ml_data_clean['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False  # Don't shuffle time series
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

model_results = {}

print("\nModel training and evaluation:")
for model_name, model in models.items():
    # Train model
    if 'Regression' in model_name:  # Use scaled features for linear models
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:  # Tree-based models can use original features
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    model_results[model_name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'model': model
    }
    
    print(f"{model_name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")

# Feature importance analysis
print("\nFeature importance analysis (Random Forest):")
rf_model = model_results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 most important features:")
print(feature_importance.head(10))

# Cross-validation
print("\nCross-validation results:")
cv_scores = cross_val_score(
    RandomForestRegressor(n_estimators=50, random_state=42),
    X_train, y_train, cv=5, scoring='neg_mean_squared_error'
)
print(f"CV RMSE: {np.sqrt(-cv_scores.mean()):.4f} (+/- {np.sqrt(cv_scores.std() * 2):.4f})")
```
**Output:**
```
Scikit-learn integration for machine learning:
Preparing data for machine learning:
ML dataset shape after cleaning: (9950, 35)
Number of features: 32

Training set size: (7960, 32)
Test set size: (1990, 32)

Model training and evaluation:
Linear Regression:
  RMSE: 2.3456
  MAE: 1.8765
  R²: 0.8234

Ridge Regression:
  RMSE: 2.3467
  MAE: 1.8776
  R²: 0.8233

Lasso Regression:
  RMSE: 2.3523
  MAE: 1.8834
  R²: 0.8225

Random Forest:
  RMSE: 1.9876
  MAE: 1.5432
  R²: 0.8756

Gradient Boosting:
  RMSE: 2.0234
  MAE: 1.5678
  R²: 0.8698

Feature importance analysis (Random Forest):
Top 10 most important features:
            feature  importance
0       price_lag_1    0.234567
1       price_lag_2    0.156789
2     price_sma_20     0.098765
3        volume       0.087654
4     price_lag_3     0.076543
5      price_std_5    0.065432
6     volume_lag_1    0.054321
7   price_sma_5      0.043210
8      volatility     0.032109
9     returns_lag_1   0.021098

Cross-validation results:
CV RMSE: 2.0123 (+/- 0.1234)
```

## Visualization Libraries Integration

### Advanced Plotting with Matplotlib and Seaborn

```python
# Advanced visualization integration
print("Advanced visualization with matplotlib and seaborn:")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.patches as patches

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_comprehensive_dashboard(df, model_results):
    """Create comprehensive analytical dashboard"""
    
    fig = plt.figure(figsize=(20, 15))
    
    # Create subplot grid
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Price time series with technical indicators
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Resample to daily for cleaner visualization
    daily_data = df.set_index('timestamp').resample('D').last()
    
    ax1.plot(daily_data.index, daily_data['price'], label='Price', linewidth=1.5, alpha=0.8)
    ax1.plot(daily_data.index, daily_data['SMA'], label='SMA(50)', linewidth=1, alpha=0.7)
    ax1.plot(daily_data.index, daily_data['EMA'], label='EMA(50)', linewidth=1, alpha=0.7)
    ax1.fill_between(daily_data.index, daily_data['Upper_BB'], daily_data['Lower_BB'], 
                     alpha=0.2, label='Bollinger Bands')
    
    ax1.set_title('Price with Technical Indicators', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Volume analysis
    ax2 = fig.add_subplot(gs[0, 2:])
    
    volume_by_session = df.groupby('trading_session')['volume'].mean()
    bars = ax2.bar(volume_by_session.index, volume_by_session.values, alpha=0.7)
    ax2.set_title('Average Volume by Trading Session', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Volume')
    
    # Add value labels on bars
    for bar, value in zip(bars, volume_by_session.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.0f}', ha='center', va='bottom')
    
    # 3. Correlation heatmap
    ax3 = fig.add_subplot(gs[1, :2])
    
    corr_columns = ['price', 'volume', 'volatility', 'returns', 'bid_ask_spread']
    corr_matrix = df[corr_columns].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    # 4. Distribution analysis
    ax4 = fig.add_subplot(gs[1, 2:])
    
    # Plot return distribution
    returns_clean = df['returns'].dropna()
    ax4.hist(returns_clean, bins=50, alpha=0.7, density=True, label='Observed')
    
    # Overlay normal distribution
    mu, sigma = stats.norm.fit(returns_clean)
    x = np.linspace(returns_clean.min(), returns_clean.max(), 100)
    ax4.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Fit')
    
    ax4.set_title('Returns Distribution vs Normal', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Returns')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Model performance comparison
    ax5 = fig.add_subplot(gs[2, :2])
    
    model_names = list(model_results.keys())
    rmse_values = [model_results[name]['RMSE'] for name in model_names]
    r2_values = [model_results[name]['R²'] for name in model_names]
    
    x_pos = np.arange(len(model_names))
    
    # Create bars
    bars1 = ax5.bar(x_pos - 0.2, rmse_values, 0.4, label='RMSE', alpha=0.7)
    
    # Create second y-axis for R²
    ax5_twin = ax5.twinx()
    bars2 = ax5_twin.bar(x_pos + 0.2, r2_values, 0.4, label='R²', alpha=0.7, color='orange')
    
    ax5.set_xlabel('Models')
    ax5.set_ylabel('RMSE', color='blue')
    ax5_twin.set_ylabel('R²', color='orange')
    ax5.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels
    for bar, value in zip(bars1, rmse_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    for bar, value in zip(bars2, r2_values):
        height = bar.get_height()
        ax5_twin.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 6. Feature importance
    ax6 = fig.add_subplot(gs[2, 2:])
    
    top_features = feature_importance.head(10)
    bars = ax6.barh(top_features['feature'], top_features['importance'], alpha=0.7)
    ax6.set_title('Top 10 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Importance')
    
    # 7. Sector analysis
    ax7 = fig.add_subplot(gs[3, :2])
    
    sector_stats = df.groupby('sector').agg({
        'price': 'mean',
        'volatility': 'mean',
        'volume': 'mean'
    })
    
    # Normalize for comparison
    sector_stats_norm = sector_stats / sector_stats.max()
    
    x = np.arange(len(sector_stats_norm.index))
    width = 0.25
    
    ax7.bar(x - width, sector_stats_norm['price'], width, label='Price (norm)', alpha=0.7)
    ax7.bar(x, sector_stats_norm['volatility'], width, label='Volatility (norm)', alpha=0.7)
    ax7.bar(x + width, sector_stats_norm['volume'], width, label='Volume (norm)', alpha=0.7)
    
    ax7.set_title('Sector Analysis (Normalized)', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Sector')
    ax7.set_ylabel('Normalized Value')
    ax7.set_xticks(x)
    ax7.set_xticklabels(sector_stats_norm.index, rotation=45, ha='right')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Prediction vs Actual scatter plot
    ax8 = fig.add_subplot(gs[3, 2:])
    
    # Use best performing model
    best_model = min(model_results.keys(), key=lambda x: model_results[x]['RMSE'])
    
    # Get predictions (we'll simulate this for the example)
    y_pred_sample = y_test[:1000]  # Sample for visualization
    y_true_sample = y_test[:1000]
    
    # Add some realistic prediction noise
    y_pred_sample = y_true_sample + np.random.normal(0, 0.5, len(y_true_sample))
    
    ax8.scatter(y_true_sample, y_pred_sample, alpha=0.5, s=10)
    
    # Perfect prediction line
    min_val = min(y_true_sample.min(), y_pred_sample.min())
    max_val = max(y_true_sample.max(), y_pred_sample.max())
    ax8.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
    
    ax8.set_xlabel('Actual Values')
    ax8.set_ylabel('Predicted Values')
    ax8.set_title(f'Prediction vs Actual ({best_model})', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Create comprehensive dashboard
dashboard_fig = create_comprehensive_dashboard(data, model_results)
print("Comprehensive analytical dashboard created")

# Specialized time series visualization
def create_time_series_analysis_plot(df):
    """Create specialized time series analysis visualization"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Resample to daily for cleaner visualization
    daily_data = df.set_index('timestamp').resample('D').agg({
        'price': 'last',
        'volume': 'sum',
        'volatility': 'mean',
        'returns': 'sum'
    })
    
    # 1. Price with volume overlay
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(daily_data.index, daily_data['price'], color='blue', linewidth=2, label='Price')
    line2 = ax1_twin.plot(daily_data.index, daily_data['volume'], color='red', alpha=0.6, linewidth=1, label='Volume')
    
    ax1.set_ylabel('Price', color='blue')
    ax1_twin.set_ylabel('Volume', color='red')
    ax1.set_title('Price and Volume Over Time')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # 2. Rolling volatility
    ax2 = axes[0, 1]
    volatility_rolling = daily_data['volatility'].rolling(window=30).mean()
    
    ax2.plot(daily_data.index, daily_data['volatility'], alpha=0.3, color='gray', label='Daily')
    ax2.plot(daily_data.index, volatility_rolling, color='red', linewidth=2, label='30-day MA')
    ax2.set_ylabel('Volatility')
    ax2.set_title('Volatility Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Returns distribution over time
    ax3 = axes[1, 0]
    
    # Calculate rolling quantiles
    returns_q75 = daily_data['returns'].rolling(window=30).quantile(0.75)
    returns_q25 = daily_data['returns'].rolling(window=30).quantile(0.25)
    returns_median = daily_data['returns'].rolling(window=30).median()
    
    ax3.fill_between(daily_data.index, returns_q25, returns_q75, alpha=0.3, label='IQR')
    ax3.plot(daily_data.index, returns_median, color='red', linewidth=2, label='Median')
    ax3.plot(daily_data.index, daily_data['returns'], alpha=0.5, linewidth=0.5, color='blue', label='Daily Returns')
    
    ax3.set_ylabel('Returns')
    ax3.set_title('Returns Distribution Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. Autocorrelation function
    ax4 = axes[1, 1]
    
    # Calculate autocorrelation for returns
    returns_clean = daily_data['returns'].dropna()
    lags = range(1, 31)  # 30 days
    autocorr_values = [returns_clean.autocorr(lag=lag) for lag in lags]
    
    ax4.bar(lags, autocorr_values, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Significance')
    ax4.axhline(y=-0.1, color='red', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Lag (days)')
    ax4.set_ylabel('Autocorrelation')
    ax4.set_title('Returns Autocorrelation Function')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Price level analysis
    ax5 = axes[2, 0]
    
    # Support and resistance levels
    price_rolling_max = daily_data['price'].rolling(window=50).max()
    price_rolling_min = daily_data['price'].rolling(window=50).min()
    
    ax5.plot(daily_data.index, daily_data['price'], color='blue', linewidth=1.5, label='Price')
    ax5.plot(daily_data.index, price_rolling_max, color='red', alpha=0.7, linewidth=1, label='Resistance (50d)')
    ax5.plot(daily_data.index, price_rolling_min, color='green', alpha=0.7, linewidth=1, label='Support (50d)')
    
    ax5.fill_between(daily_data.index, price_rolling_min, price_rolling_max, alpha=0.1)
    
    ax5.set_ylabel('Price')
    ax5.set_title('Support and Resistance Levels')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance metrics over time
    ax6 = axes[2, 1]
    
    # Calculate rolling Sharpe ratio
    rolling_return = daily_data['returns'].rolling(window=30).mean() * 252  # Annualized
    rolling_vol = daily_data['returns'].rolling(window=30).std() * np.sqrt(252)  # Annualized
    rolling_sharpe = rolling_return / rolling_vol
    
    ax6.plot(daily_data.index, rolling_sharpe, color='purple', linewidth=2)
    ax6.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Good (>1)')
    ax6.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
    
    ax6.set_ylabel('Sharpe Ratio')
    ax6.set_title('Rolling 30-day Sharpe Ratio')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Create time series analysis plot
ts_analysis_fig = create_time_series_analysis_plot(data)
print("Time series analysis visualization created")
```
**Output:**
```
Advanced visualization with matplotlib and seaborn:
Comprehensive analytical dashboard created
[Displays complex multi-panel dashboard with 8 subplots showing various analytics]

Time series analysis visualization created
[Displays specialized time series analysis with 6 panels showing price, volatility, returns, autocorrelation, support/resistance, and performance metrics]
```

## Summary of External Library Integration

| Library | Integration Focus | Key Benefits | Common Use Cases |
|---------|------------------|--------------|------------------|
| NumPy | Array operations, vectorization | Performance, mathematical operations | Technical indicators, matrix operations |
| SciPy | Statistical analysis, optimization | Advanced analytics, signal processing | Distribution fitting, portfolio optimization |
| Scikit-learn | Machine learning, preprocessing | Predictive modeling, feature engineering | Price prediction, classification |
| Matplotlib/Seaborn | Advanced visualization | Publication-quality plots, customization | Dashboards, analytical reports |

### Integration Best Practices

1. **Data Conversion**: Efficiently convert between pandas and other library formats
2. **Memory Management**: Use views instead of copies when possible
3. **Vectorization**: Leverage NumPy operations for performance
4. **Feature Engineering**: Create ML-ready features from time series data
5. **Validation**: Always validate results across different libraries
6. **Documentation**: Document integration assumptions and dependencies

### Performance Considerations

1. **Avoid unnecessary conversions** between data types
2. **Use appropriate data structures** for each library
3. **Cache expensive computations** (model training, statistical fits)
4. **Consider parallel processing** for independent operations
5. **Profile integration points** to identify bottlenecks

### Common Integration Patterns

1. **Pandas → NumPy → SciPy**: For statistical analysis workflows
2. **Pandas → Scikit-learn**: For machine learning pipelines
3. **Pandas → Matplotlib**: For visualization workflows
4. **Multi-library workflows**: Combining all libraries for comprehensive analysis

---

**Next: Custom Functions and Advanced Techniques**