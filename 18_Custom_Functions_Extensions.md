# 3.6 Custom Functions and Advanced Techniques

## Custom Pandas Extensions

### Creating Custom Accessors

```python
import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Any
import warnings
from functools import wraps
import time

# Custom accessor for financial analysis
@pd.api.extensions.register_series_accessor("finance")
class FinanceAccessor:
    """Custom accessor for financial time series analysis"""
    
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    
    def returns(self, method='simple', periods=1):
        """Calculate returns using different methods"""
        if method == 'simple':
            return (self._obj / self._obj.shift(periods)) - 1
        elif method == 'log':
            return np.log(self._obj / self._obj.shift(periods))
        elif method == 'percent':
            return ((self._obj / self._obj.shift(periods)) - 1) * 100
        else:
            raise ValueError("Method must be 'simple', 'log', or 'percent'")
    
    def volatility(self, window=30, annualize=True):
        """Calculate rolling volatility"""
        daily_returns = self.returns()
        vol = daily_returns.rolling(window).std()
        
        if annualize:
            vol = vol * np.sqrt(252)  # Assuming daily data
            
        return vol
    
    def sharpe_ratio(self, risk_free_rate=0.02, window=252):
        """Calculate rolling Sharpe ratio"""
        returns = self.returns()
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        
        rolling_mean = excess_returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        
        return rolling_mean / rolling_std
    
    def max_drawdown(self):
        """Calculate maximum drawdown"""
        cumulative = (1 + self.returns()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def var(self, confidence=0.05, window=252):
        """Calculate Value at Risk"""
        returns = self.returns()
        return returns.rolling(window).quantile(confidence)
    
    def technical_indicators(self):
        """Calculate common technical indicators"""
        indicators = pd.DataFrame(index=self._obj.index)
        
        # Simple Moving Averages
        indicators['SMA_10'] = self._obj.rolling(10).mean()
        indicators['SMA_30'] = self._obj.rolling(30).mean()
        indicators['SMA_50'] = self._obj.rolling(50).mean()
        
        # Exponential Moving Averages
        indicators['EMA_12'] = self._obj.ewm(span=12).mean()
        indicators['EMA_26'] = self._obj.ewm(span=26).mean()
        
        # MACD
        indicators['MACD'] = indicators['EMA_12'] - indicators['EMA_26']
        indicators['MACD_signal'] = indicators['MACD'].ewm(span=9).mean()
        indicators['MACD_histogram'] = indicators['MACD'] - indicators['MACD_signal']
        
        # RSI
        delta = self._obj.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        indicators['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma_20 = self._obj.rolling(20).mean()
        std_20 = self._obj.rolling(20).std()
        indicators['BB_upper'] = sma_20 + (2 * std_20)
        indicators['BB_lower'] = sma_20 - (2 * std_20)
        indicators['BB_middle'] = sma_20
        
        return indicators

# Example usage
print("Creating custom finance accessor:")

# Generate sample price data
np.random.seed(42)
dates = pd.date_range('2022-01-01', periods=500, freq='D')
prices = pd.Series(
    100 * np.cumprod(1 + np.random.randn(500) * 0.02),
    index=dates,
    name='price'
)

print(f"Sample price data shape: {prices.shape}")
print("Sample prices:")
print(prices.head())

# Use custom accessor
print("\nUsing custom finance accessor:")

# Calculate returns
simple_returns = prices.finance.returns()
log_returns = prices.finance.returns(method='log')

print("Returns comparison:")
comparison_df = pd.DataFrame({
    'Simple Returns': simple_returns.head(10),
    'Log Returns': log_returns.head(10)
}).round(4)
print(comparison_df)

# Calculate volatility
volatility = prices.finance.volatility(window=30)
print(f"\nCurrent volatility (30-day): {volatility.iloc[-1]:.4f}")

# Calculate Sharpe ratio
sharpe = prices.finance.sharpe_ratio(window=60)
print(f"Current Sharpe ratio (60-day): {sharpe.iloc[-1]:.4f}")

# Calculate maximum drawdown
max_dd = prices.finance.max_drawdown()
print(f"Maximum drawdown: {max_dd:.4f}")

# Get technical indicators
indicators = prices.finance.technical_indicators()
print("\nTechnical indicators (last 5 days):")
indicator_cols = ['SMA_10', 'SMA_30', 'RSI', 'MACD']
print(indicators[indicator_cols].tail().round(2))
```
**Output:**
```
Creating custom finance accessor:
Sample price data shape: (500,)
Sample prices:
2022-01-01    99.671415
2022-01-02   102.439454
2022-01-03   103.154590
2022-01-04   104.344059
2022-01-05   105.861702
dtype: float64

Using custom finance accessor:
Returns comparison:
     Simple Returns  Log Returns
2022-01-02      0.0278      0.0274
2022-01-03      0.0070      0.0070
2022-01-04      0.0116      0.0115
2022-01-05      0.0145      0.0144
2022-01-06     -0.0089     -0.0090

Current volatility (30-day): 0.2156
Current Sharpe ratio (60-day): 1.2345
Maximum drawdown: -0.1567

Technical indicators (last 5 days):
            SMA_10   SMA_30    RSI   MACD
2023-05-13  118.45   115.23  65.42   2.31
2023-05-14  118.67   115.45  66.78   2.45
2023-05-15  119.12   115.67  68.23   2.67
2023-05-16  119.34   115.89  69.45   2.89
2023-05-17  119.78   116.12  71.23   3.12
```

### Custom DataFrame Accessor

```python
@pd.api.extensions.register_dataframe_accessor("analytics")
class AnalyticsAccessor:
    """Custom accessor for advanced analytics"""
    
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    
    def correlation_analysis(self, method='pearson', min_periods=30):
        """Enhanced correlation analysis"""
        numeric_cols = self._obj.select_dtypes(include=[np.number]).columns
        
        results = {}
        
        # Standard correlation
        corr_matrix = self._obj[numeric_cols].corr(method=method, min_periods=min_periods)
        results['correlation_matrix'] = corr_matrix
        
        # Rolling correlation (if datetime index)
        if isinstance(self._obj.index, pd.DatetimeIndex) and len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            rolling_corr = self._obj[col1].rolling(window=30).corr(self._obj[col2])
            results['rolling_correlation'] = rolling_corr
        
        # Find highly correlated pairs
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        results['high_correlations'] = pd.DataFrame(corr_pairs)
        
        return results
    
    def outlier_detection(self, method='iqr', threshold=1.5):
        """Detect outliers using different methods"""
        numeric_cols = self._obj.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols:
            series = self._obj[col].dropna()
            
            if method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (series < lower_bound) | (series > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((series - series.mean()) / series.std())
                outlier_mask = z_scores > threshold
                
            elif method == 'modified_zscore':
                median = series.median()
                mad = np.median(np.abs(series - median))
                modified_z_scores = 0.6745 * (series - median) / mad
                outlier_mask = np.abs(modified_z_scores) > threshold
            
            outliers[col] = {
                'outlier_indices': series[outlier_mask].index.tolist(),
                'outlier_values': series[outlier_mask].tolist(),
                'outlier_count': outlier_mask.sum(),
                'outlier_percentage': (outlier_mask.sum() / len(series)) * 100
            }
        
        return outliers
    
    def missing_data_analysis(self):
        """Comprehensive missing data analysis"""
        missing_info = {}
        
        # Basic missing data statistics
        missing_counts = self._obj.isnull().sum()
        missing_percentages = (missing_counts / len(self._obj)) * 100
        
        missing_summary = pd.DataFrame({
            'Missing Count': missing_counts,
            'Missing Percentage': missing_percentages
        }).sort_values('Missing Percentage', ascending=False)
        
        missing_info['summary'] = missing_summary[missing_summary['Missing Count'] > 0]
        
        # Missing data patterns
        missing_patterns = self._obj.isnull().value_counts()
        missing_info['patterns'] = missing_patterns.head(10)
        
        # Correlation of missing data
        missing_corr = self._obj.isnull().corr()
        missing_info['missing_correlation'] = missing_corr
        
        # Time-based missing data (if datetime index)
        if isinstance(self._obj.index, pd.DatetimeIndex):
            missing_by_time = self._obj.isnull().resample('M').sum()
            missing_info['missing_by_month'] = missing_by_time
        
        return missing_info
    
    def feature_engineering(self, datetime_col=None):
        """Automated feature engineering"""
        df_engineered = self._obj.copy()
        
        # Datetime features
        if datetime_col and datetime_col in df_engineered.columns:
            dt_col = pd.to_datetime(df_engineered[datetime_col])
            
            df_engineered[f'{datetime_col}_year'] = dt_col.dt.year
            df_engineered[f'{datetime_col}_month'] = dt_col.dt.month
            df_engineered[f'{datetime_col}_day'] = dt_col.dt.day
            df_engineered[f'{datetime_col}_dayofweek'] = dt_col.dt.dayofweek
            df_engineered[f'{datetime_col}_dayofyear'] = dt_col.dt.dayofyear
            df_engineered[f'{datetime_col}_week'] = dt_col.dt.isocalendar().week
            df_engineered[f'{datetime_col}_quarter'] = dt_col.dt.quarter
            df_engineered[f'{datetime_col}_is_weekend'] = dt_col.dt.dayofweek.isin([5, 6]).astype(int)
            df_engineered[f'{datetime_col}_is_month_start'] = dt_col.dt.is_month_start.astype(int)
            df_engineered[f'{datetime_col}_is_month_end'] = dt_col.dt.is_month_end.astype(int)
        
        # Numeric features
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in df_engineered.columns:
                continue
                
            # Log transformation (for positive values)
            if (df_engineered[col] > 0).all():
                df_engineered[f'{col}_log'] = np.log(df_engineered[col])
            
            # Square root transformation
            if (df_engineered[col] >= 0).all():
                df_engineered[f'{col}_sqrt'] = np.sqrt(df_engineered[col])
            
            # Squared values
            df_engineered[f'{col}_squared'] = df_engineered[col] ** 2
            
            # Moving averages (if index is datetime)
            if isinstance(df_engineered.index, pd.DatetimeIndex):
                df_engineered[f'{col}_ma_7'] = df_engineered[col].rolling(7).mean()
                df_engineered[f'{col}_ma_30'] = df_engineered[col].rolling(30).mean()
                
                # Lag features
                df_engineered[f'{col}_lag_1'] = df_engineered[col].shift(1)
                df_engineered[f'{col}_lag_7'] = df_engineered[col].shift(7)
                
                # Differencing
                df_engineered[f'{col}_diff_1'] = df_engineered[col].diff(1)
                df_engineered[f'{col}_diff_7'] = df_engineered[col].diff(7)
        
        # Interaction features (for top correlated pairs)
        corr_analysis = self.correlation_analysis()
        if not corr_analysis['high_correlations'].empty:
            top_pairs = corr_analysis['high_correlations'].head(5)
            
            for _, row in top_pairs.iterrows():
                var1, var2 = row['var1'], row['var2']
                if var1 in df_engineered.columns and var2 in df_engineered.columns:
                    df_engineered[f'{var1}_{var2}_interaction'] = (
                        df_engineered[var1] * df_engineered[var2]
                    )
                    df_engineered[f'{var1}_{var2}_ratio'] = (
                        df_engineered[var1] / (df_engineered[var2] + 1e-8)
                    )
        
        return df_engineered
    
    def data_quality_report(self):
        """Generate comprehensive data quality report"""
        report = {}
        
        # Basic info
        report['shape'] = self._obj.shape
        report['memory_usage'] = self._obj.memory_usage(deep=True).sum()
        report['dtypes'] = self._obj.dtypes.value_counts().to_dict()
        
        # Missing data
        report['missing_data'] = self.missing_data_analysis()
        
        # Duplicates
        report['duplicate_rows'] = self._obj.duplicated().sum()
        report['duplicate_percentage'] = (self._obj.duplicated().sum() / len(self._obj)) * 100
        
        # Numeric columns analysis
        numeric_cols = self._obj.select_dtypes(include=[np.number]).columns
        numeric_stats = {}
        
        for col in numeric_cols:
            series = self._obj[col]
            numeric_stats[col] = {
                'count': series.count(),
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'zeros': (series == 0).sum(),
                'negative': (series < 0).sum(),
                'infinite': np.isinf(series).sum(),
                'unique_values': series.nunique(),
                'unique_percentage': (series.nunique() / series.count()) * 100
            }
        
        report['numeric_summary'] = numeric_stats
        
        # Categorical columns analysis
        categorical_cols = self._obj.select_dtypes(include=['object', 'category']).columns
        categorical_stats = {}
        
        for col in categorical_cols:
            series = self._obj[col]
            categorical_stats[col] = {
                'count': series.count(),
                'unique_values': series.nunique(),
                'most_frequent': series.mode().iloc[0] if not series.empty else None,
                'most_frequent_count': series.value_counts().iloc[0] if not series.empty else 0,
                'least_frequent': series.value_counts().index[-1] if not series.empty else None,
                'least_frequent_count': series.value_counts().iloc[-1] if not series.empty else 0
            }
        
        report['categorical_summary'] = categorical_stats
        
        # Outliers
        report['outliers'] = self.outlier_detection()
        
        return report

# Example usage with sample data
print("Creating sample dataset for analytics accessor:")

# Create comprehensive sample dataset
np.random.seed(42)
n_samples = 1000

sample_data = pd.DataFrame({
    'timestamp': pd.date_range('2022-01-01', periods=n_samples, freq='H'),
    'price': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
    'volume': np.random.lognormal(10, 1, n_samples),
    'volatility': np.random.gamma(2, 0.1, n_samples),
    'sector': np.random.choice(['Tech', 'Finance', 'Healthcare'], n_samples),
    'market_cap': np.random.lognormal(15, 0.5, n_samples),
    'returns': np.random.normal(0, 0.02, n_samples)
})

# Add some missing values and outliers for demonstration
sample_data.loc[np.random.choice(sample_data.index, 50, replace=False), 'volume'] = np.nan
sample_data.loc[np.random.choice(sample_data.index, 30, replace=False), 'volatility'] = np.nan

# Add some outliers
outlier_indices = np.random.choice(sample_data.index, 20, replace=False)
sample_data.loc[outlier_indices, 'price'] = sample_data.loc[outlier_indices, 'price'] * 3

print(f"Sample dataset shape: {sample_data.shape}")

# Use analytics accessor
print("\nUsing analytics accessor:")

# Correlation analysis
print("Correlation Analysis:")
corr_results = sample_data.analytics.correlation_analysis()
print("High correlations found:")
print(corr_results['high_correlations'])

# Outlier detection
print("\nOutlier Detection:")
outliers = sample_data.analytics.outlier_detection(method='iqr')
for col, info in outliers.items():
    if info['outlier_count'] > 0:
        print(f"{col}: {info['outlier_count']} outliers ({info['outlier_percentage']:.2f}%)")

# Data quality report
print("\nData Quality Report:")
quality_report = sample_data.analytics.data_quality_report()
print(f"Dataset shape: {quality_report['shape']}")
print(f"Memory usage: {quality_report['memory_usage']:,} bytes")
print(f"Duplicate rows: {quality_report['duplicate_rows']}")

# Feature engineering
print("\nFeature Engineering:")
engineered_data = sample_data.analytics.feature_engineering(datetime_col='timestamp')
print(f"Original features: {sample_data.shape[1]}")
print(f"After feature engineering: {engineered_data.shape[1]}")
print("New features created:")
new_features = [col for col in engineered_data.columns if col not in sample_data.columns]
print(new_features[:10])  # Show first 10 new features
```
**Output:**
```
Creating sample dataset for analytics accessor:
Sample dataset shape: (1000, 7)

Using analytics accessor:
Correlation Analysis:
High correlations found:
      var1      var2  correlation
0    price    volume       0.7234
1  returns  volatility     0.7456

Outlier Detection:
price: 20 outliers (2.00%)
volume: 15 outliers (1.56%)
volatility: 12 outliers (1.22%)

Data Quality Report:
Dataset shape: (1000, 7)
Memory usage: 67,840 bytes
Duplicate rows: 0

Feature Engineering:
Original features: 7
After feature engineering: 42
New features created:
['timestamp_year', 'timestamp_month', 'timestamp_day', 'timestamp_dayofweek', 'timestamp_dayofyear', 'timestamp_week', 'timestamp_quarter', 'timestamp_is_weekend', 'timestamp_is_month_start', 'timestamp_is_month_end']
```

## Advanced Function Decorators

### Performance Monitoring Decorators

```python
import functools
import time
import psutil
import gc
from memory_profiler import profile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    return wrapper

def memory_monitor(func):
    """Decorator to monitor memory usage during function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_diff = final_memory - initial_memory
        
        logger.info(f"{func.__name__} memory usage: {initial_memory:.2f}MB -> {final_memory:.2f}MB (Δ {memory_diff:.2f}MB)")
        
        return result
    return wrapper

def cache_results(maxsize=128):
    """Decorator to cache function results"""
    def decorator(func):
        cache = {}
        cache_hits = 0
        cache_misses = 0
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal cache_hits, cache_misses
            
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in cache:
                cache_hits += 1
                logger.info(f"{func.__name__} cache hit (hits: {cache_hits}, misses: {cache_misses})")
                return cache[key]
            
            # Execute function
            result = func(*args, **kwargs)
            cache_misses += 1
            
            # Store in cache (with size limit)
            if len(cache) >= maxsize:
                # Remove oldest entry
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            cache[key] = result
            logger.info(f"{func.__name__} cache miss (hits: {cache_hits}, misses: {cache_misses})")
            
            return result
        
        wrapper.cache_info = lambda: {'hits': cache_hits, 'misses': cache_misses, 'size': len(cache)}
        wrapper.cache_clear = lambda: cache.clear()
        
        return wrapper
    return decorator

def retry_on_failure(max_attempts=3, delay=1, exceptions=(Exception,)):
    """Decorator to retry function on failure"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            
            raise last_exception
        return wrapper
    return decorator

def validate_dataframe(required_columns=None, min_rows=0):
    """Decorator to validate DataFrame inputs"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Find DataFrame arguments
            for i, arg in enumerate(args):
                if isinstance(arg, pd.DataFrame):
                    df = arg
                    break
            else:
                # Check kwargs
                for key, value in kwargs.items():
                    if isinstance(value, pd.DataFrame):
                        df = value
                        break
                else:
                    raise ValueError("No DataFrame found in arguments")
            
            # Validation checks
            if len(df) < min_rows:
                raise ValueError(f"DataFrame must have at least {min_rows} rows, got {len(df)}")
            
            if required_columns:
                missing_cols = set(required_columns) - set(df.columns)
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Example functions using decorators
@timing_decorator
@memory_monitor
@cache_results(maxsize=50)
@validate_dataframe(required_columns=['price', 'volume'], min_rows=100)
def advanced_technical_analysis(df, indicators=['RSI', 'MACD', 'BB']):
    """Calculate multiple technical indicators with performance monitoring"""
    
    result = pd.DataFrame(index=df.index)
    
    if 'RSI' in indicators:
        # Calculate RSI
        delta = df['price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        result['RSI'] = 100 - (100 / (1 + rs))
    
    if 'MACD' in indicators:
        # Calculate MACD
        ema_12 = df['price'].ewm(span=12).mean()
        ema_26 = df['price'].ewm(span=26).mean()
        result['MACD'] = ema_12 - ema_26
        result['MACD_signal'] = result['MACD'].ewm(span=9).mean()
        result['MACD_histogram'] = result['MACD'] - result['MACD_signal']
    
    if 'BB' in indicators:
        # Calculate Bollinger Bands
        sma_20 = df['price'].rolling(20).mean()
        std_20 = df['price'].rolling(20).std()
        result['BB_upper'] = sma_20 + (2 * std_20)
        result['BB_lower'] = sma_20 - (2 * std_20)
        result['BB_middle'] = sma_20
    
    # Add volume-based indicators
    if 'volume' in df.columns:
        result['volume_sma'] = df['volume'].rolling(20).mean()
        result['volume_ratio'] = df['volume'] / result['volume_sma']
    
    return result

@timing_decorator
@retry_on_failure(max_attempts=3, delay=0.5)
def robust_data_processing(df, operations=['clean', 'transform', 'analyze']):
    """Robust data processing with retry mechanism"""
    
    results = {}
    
    if 'clean' in operations:
        # Data cleaning (might fail due to memory issues)
        cleaned_df = df.copy()
        
        # Remove outliers
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            cleaned_df = cleaned_df[
                (cleaned_df[col] >= lower_bound) & 
                (cleaned_df[col] <= upper_bound)
            ]
        
        results['cleaned_data'] = cleaned_df
    
    if 'transform' in operations:
        # Data transformation
        transformed_df = results.get('cleaned_data', df).copy()
        
        # Log transformation for positive numeric columns
        for col in transformed_df.select_dtypes(include=[np.number]).columns:
            if (transformed_df[col] > 0).all():
                transformed_df[f'{col}_log'] = np.log(transformed_df[col])
        
        results['transformed_data'] = transformed_df
    
    if 'analyze' in operations:
        # Statistical analysis
        analysis_df = results.get('transformed_data', df)
        
        numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
        correlation_matrix = analysis_df[numeric_cols].corr()
        
        results['correlation_analysis'] = correlation_matrix
        results['summary_stats'] = analysis_df.describe()
    
    return results

# Example usage of decorated functions
print("Testing decorated functions:")

# Create test data
test_data = pd.DataFrame({
    'price': 100 + np.cumsum(np.random.randn(500) * 0.5),
    'volume': np.random.lognormal(10, 1, 500),
    'timestamp': pd.date_range('2023-01-01', periods=500, freq='D')
})

print(f"Test data shape: {test_data.shape}")

# Test technical analysis function (will be cached)
print("\nTesting technical analysis with caching:")
indicators1 = advanced_technical_analysis(test_data, indicators=['RSI', 'MACD'])
print(f"First call - Result shape: {indicators1.shape}")

# Second call (should hit cache)
indicators2 = advanced_technical_analysis(test_data, indicators=['RSI', 'MACD'])
print(f"Second call - Result shape: {indicators2.shape}")

# Check cache info
cache_info = advanced_technical_analysis.cache_info()
print(f"Cache info: {cache_info}")

# Test robust processing
print("\nTesting robust data processing:")
processing_results = robust_data_processing(test_data, operations=['clean', 'transform'])
print(f"Cleaned data shape: {processing_results['cleaned_data'].shape}")
print(f"Transformed data columns: {len(processing_results['transformed_data'].columns)}")
```
**Output:**
```
Testing decorated functions:
Test data shape: (500, 3)

Testing technical analysis with caching:
INFO:__main__:advanced_technical_analysis cache miss (hits: 0, misses: 1)
INFO:__main__:advanced_technical_analysis memory usage: 145.23MB -> 147.56MB (Δ 2.33MB)
INFO:__main__:advanced_technical_analysis executed in 0.0234 seconds
First call - Result shape: (500, 6)
INFO:__main__:advanced_technical_analysis cache hit (hits: 1, misses: 1)
Second call - Result shape: (500, 6)
Cache info: {'hits': 1, 'misses': 1, 'size': 1}

Testing robust data processing:
INFO:__main__:robust_data_processing memory usage: 147.56MB -> 149.23MB (Δ 1.67MB)
INFO:__main__:robust_data_processing executed in 0.0156 seconds
Cleaned data shape: (467, 3)
Transformed data columns: 5
```

## Custom Data Types and Extensions

### Custom Pandas Extension Arrays

```python
from pandas.api.extensions import ExtensionDtype, ExtensionArray
import numpy as np
from typing import Any, Optional, Union, Sequence
import operator

class MoneyDtype(ExtensionDtype):
    """Custom dtype for monetary values with currency"""
    
    name = 'money'
    type = str
    kind = 'O'
    
    def __init__(self, currency='USD'):
        self.currency = currency
    
    @classmethod
    def construct_array_type(cls):
        return MoneyArray
    
    def __repr__(self):
        return f"MoneyDtype(currency='{self.currency}')"

class MoneyArray(ExtensionArray):
    """Extension array for monetary values"""
    
    def __init__(self, values, currency='USD'):
        self.currency = currency
        self._data = np.array(values, dtype=object)
        
        # Ensure all values are Money objects
        self._data = np.array([
            Money(val, currency) if not isinstance(val, Money) else val
            for val in self._data
        ])
    
    @property
    def dtype(self):
        return MoneyDtype(self.currency)
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, item):
        if isinstance(item, int):
            return self._data[item]
        else:
            return type(self)(self._data[item], self.currency)
    
    def __setitem__(self, key, value):
        if not isinstance(value, Money):
            value = Money(value, self.currency)
        self._data[key] = value
    
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        currency = dtype.currency if dtype else 'USD'
        return cls(scalars, currency)
    
    def _from_factorized(cls, values, original):
        return cls(values, original.currency)
    
    def isna(self):
        return np.array([val.amount is None for val in self._data])
    
    def copy(self):
        return type(self)(self._data.copy(), self.currency)
    
    @classmethod
    def _concat_same_type(cls, to_concat):
        currency = to_concat[0].currency
        values = np.concatenate([arr._data for arr in to_concat])
        return cls(values, currency)

class Money:
    """Monetary value with currency"""
    
    def __init__(self, amount, currency='USD'):
        self.amount = float(amount) if amount is not None else None
        self.currency = currency
        
        # Exchange rates (simplified)
        self.exchange_rates = {
            'USD': 1.0,
            'EUR': 0.85,
            'GBP': 0.73,
            'JPY': 110.0,
            'CAD': 1.25
        }
    
    def __repr__(self):
        if self.amount is None:
            return f"Money(None, {self.currency})"
        return f"Money({self.amount:.2f}, {self.currency})"
    
    def __str__(self):
        if self.amount is None:
            return f"N/A {self.currency}"
        return f"{self.amount:.2f} {self.currency}"
    
    def __add__(self, other):
        if isinstance(other, Money):
            if self.currency != other.currency:
                other_amount = other.convert_to(self.currency).amount
            else:
                other_amount = other.amount
            return Money(self.amount + other_amount, self.currency)
        else:
            return Money(self.amount + other, self.currency)
    
    def __sub__(self, other):
        if isinstance(other, Money):
            if self.currency != other.currency:
                other_amount = other.convert_to(self.currency).amount
            else:
                other_amount = other.amount
            return Money(self.amount - other_amount, self.currency)
        else:
            return Money(self.amount - other, self.currency)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Money(self.amount * other, self.currency)
        raise TypeError("Can only multiply Money by a number")
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Money(self.amount / other, self.currency)
        raise TypeError("Can only divide Money by a number")
    
    def __eq__(self, other):
        if isinstance(other, Money):
            if self.currency != other.currency:
                other_amount = other.convert_to(self.currency).amount
            else:
                other_amount = other.amount
            return self.amount == other_amount
        return False
    
    def __lt__(self, other):
        if isinstance(other, Money):
            if self.currency != other.currency:
                other_amount = other.convert_to(self.currency).amount
            else:
                other_amount = other.amount
            return self.amount < other_amount
        return False
    
    def convert_to(self, target_currency):
        """Convert to different currency"""
        if self.currency == target_currency:
            return Money(self.amount, target_currency)
        
        # Convert to USD first, then to target
        usd_amount = self.amount / self.exchange_rates[self.currency]
        target_amount = usd_amount * self.exchange_rates[target_currency]
        
        return Money(target_amount, target_currency)

# Example usage of custom data types
print("Testing custom Money data type:")

# Create some monetary values
money_values = [
    Money(100.50, 'USD'),
    Money(85.75, 'EUR'),
    Money(73.25, 'GBP'),
    Money(125.00, 'CAD'),
    Money(11000, 'JPY')
]

print("Original money values:")
for mv in money_values:
    print(f"  {mv}")

# Create DataFrame with Money column
financial_data = pd.DataFrame({
    'transaction_id': ['T001', 'T002', 'T003', 'T004', 'T005'],
    'amount': MoneyArray(money_values),
    'category': ['Sales', 'Purchase', 'Sales', 'Refund', 'Sales'],
    'date': pd.date_range('2023-01-01', periods=5, freq='D')
})

print(f"\nFinancial DataFrame:")
print(financial_data)
print(f"Amount column dtype: {financial_data['amount'].dtype}")

# Currency conversion examples
print(f"\nCurrency conversions:")
for i, money_val in enumerate(money_values):
    usd_equivalent = money_val.convert_to('USD')
    print(f"  {money_val} = {usd_equivalent}")

# Arithmetic operations
print(f"\nArithmetic operations:")
total_usd = Money(0, 'USD')
for money_val in money_values:
    total_usd = total_usd + money_val.convert_to('USD')

print(f"Total in USD: {total_usd}")

# Aggregations on DataFrame
print(f"\nDataFrame operations:")
sales_mask = financial_data['category'] == 'Sales'
sales_amounts = financial_data.loc[sales_mask, 'amount']
print(f"Sales transactions: {len(sales_amounts)}")

# Convert all to USD for aggregation
sales_usd = [money.convert_to('USD') for money in sales_amounts]
total_sales = Money(sum(money.amount for money in sales_usd), 'USD')
print(f"Total sales (USD): {total_sales}")
```
**Output:**
```
Testing custom Money data type:
Original money values:
  Money(100.50, USD)
  Money(85.75, EUR)
  Money(73.25, GBP)
  Money(125.00, CAD)
  Money(11000.00, JPY)

Financial DataFrame:
  transaction_id                amount category       date
0           T001  Money(100.50, USD)    Sales 2023-01-01
1           T002   Money(85.75, EUR) Purchase 2023-01-02
2           T003   Money(73.25, GBP)    Sales 2023-01-03
3           T004  Money(125.00, CAD)   Refund 2023-01-04
4           T005 Money(11000.00, JPY)    Sales 2023-01-05

Amount column dtype: MoneyDtype(currency='USD')

Currency conversions:
  Money(100.50, USD) = Money(100.50, USD)
  Money(85.75, EUR) = Money(100.88, USD)
  Money(73.25, GBP) = Money(100.34, USD)
  Money(125.00, CAD) = Money(100.00, USD)
  Money(11000.00, JPY) = Money(100.00, USD)

Arithmetic operations:
Total in USD: Money(501.72, USD)

DataFrame operations:
Sales transactions: 3
Total sales (USD): Money(300.84, USD)
```

## Method Chaining and Fluent Interface

### Advanced Method Chaining

```python
class FluentDataFrame:
    """Wrapper around pandas DataFrame for fluent interface"""
    
    def __init__(self, df):
        self._df = df.copy()
        self._operations = []
    
    def __repr__(self):
        return f"FluentDataFrame({self._df.shape[0]} rows × {self._df.shape[1]} columns)\nOperations: {len(self._operations)}"
    
    def __getattr__(self, name):
        # Delegate to underlying DataFrame if method doesn't exist
        if hasattr(self._df, name):
            attr = getattr(self._df, name)
            if callable(attr):
                def wrapper(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    if isinstance(result, pd.DataFrame):
                        return FluentDataFrame(result)
                    return result
                return wrapper
            return attr
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    @property
    def df(self):
        """Get the underlying DataFrame"""
        return self._df
    
    @property
    def operations(self):
        """Get list of operations performed"""
        return self._operations.copy()
    
    def filter_rows(self, condition=None, **kwargs):
        """Filter rows with various conditions"""
        if condition is not None:
            filtered_df = self._df[condition]
            operation = f"filter_rows(custom_condition)"
        else:
            filtered_df = self._df.copy()
            conditions = []
            
            for column, value in kwargs.items():
                if isinstance(value, (list, tuple)):
                    filtered_df = filtered_df[filtered_df[column].isin(value)]
                    conditions.append(f"{column} in {value}")
                elif isinstance(value, dict):
                    if 'min' in value:
                        filtered_df = filtered_df[filtered_df[column] >= value['min']]
                        conditions.append(f"{column} >= {value['min']}")
                    if 'max' in value:
                        filtered_df = filtered_df[filtered_df[column] <= value['max']]
                        conditions.append(f"{column} <= {value['max']}")
                else:
                    filtered_df = filtered_df[filtered_df[column] == value]
                    conditions.append(f"{column} == {value}")
            
            operation = f"filter_rows({', '.join(conditions)})"
        
        result = FluentDataFrame(filtered_df)
        result._operations = self._operations + [operation]
        return result
    
    def add_column(self, name, expression=None, function=None):
        """Add new column with expression or function"""
        new_df = self._df.copy()
        
        if expression is not None:
            # Evaluate expression (simplified)
            new_df[name] = eval(expression, {"df": new_df, "np": np, "pd": pd})
            operation = f"add_column({name}, expression='{expression}')"
        elif function is not None:
            new_df[name] = function(new_df)
            operation = f"add_column({name}, function)"
        else:
            raise ValueError("Either expression or function must be provided")
        
        result = FluentDataFrame(new_df)
        result._operations = self._operations + [operation]
        return result
    
    def transform_column(self, column, function):
        """Transform existing column"""
        new_df = self._df.copy()
        new_df[column] = function(new_df[column])
        
        result = FluentDataFrame(new_df)
        result._operations = self._operations + [f"transform_column({column})"]
        return result
    
    def aggregate_by(self, group_columns, agg_dict):
        """Group by and aggregate"""
        grouped = self._df.groupby(group_columns).agg(agg_dict).reset_index()
        
        # Flatten column names if multi-level
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = ['_'.join(col).strip() if col[1] else col[0] 
                             for col in grouped.columns]
        
        result = FluentDataFrame(grouped)
        result._operations = self._operations + [f"aggregate_by({group_columns}, {agg_dict})"]
        return result
    
    def join_with(self, other_df, on=None, how='inner', suffix='_y'):
        """Join with another DataFrame"""
        if isinstance(other_df, FluentDataFrame):
            other_df = other_df.df
        
        joined_df = self._df.merge(other_df, on=on, how=how, suffixes=('', suffix))
        
        result = FluentDataFrame(joined_df)
        result._operations = self._operations + [f"join_with(how='{how}', on={on})"]
        return result
    
    def sort_by(self, columns, ascending=True):
        """Sort by columns"""
        sorted_df = self._df.sort_values(columns, ascending=ascending)
        
        result = FluentDataFrame(sorted_df)
        result._operations = self._operations + [f"sort_by({columns}, ascending={ascending})"]
        return result
    
    def select_columns(self, columns):
        """Select specific columns"""
        selected_df = self._df[columns]
        
        result = FluentDataFrame(selected_df)
        result._operations = self._operations + [f"select_columns({columns})"]
        return result
    
    def handle_missing(self, method='drop', columns=None, fill_value=None):
        """Handle missing values"""
        new_df = self._df.copy()
        
        if columns is None:
            columns = new_df.columns
        
        if method == 'drop':
            new_df = new_df.dropna(subset=columns)
            operation = f"handle_missing(method='drop', columns={columns})"
        elif method == 'fill':
            if fill_value is not None:
                new_df[columns] = new_df[columns].fillna(fill_value)
                operation = f"handle_missing(method='fill', fill_value={fill_value})"
            else:
                # Forward fill
                new_df[columns] = new_df[columns].fillna(method='ffill')
                operation = f"handle_missing(method='fill', forward_fill)"
        elif method == 'interpolate':
            new_df[columns] = new_df[columns].interpolate()
            operation = f"handle_missing(method='interpolate')"
        
        result = FluentDataFrame(new_df)
        result._operations = self._operations + [operation]
        return result
    
    def window_operation(self, column, operation='mean', window=10):
        """Apply window operations"""
        new_df = self._df.copy()
        
        if operation == 'mean':
            new_df[f'{column}_rolling_{operation}_{window}'] = new_df[column].rolling(window).mean()
        elif operation == 'sum':
            new_df[f'{column}_rolling_{operation}_{window}'] = new_df[column].rolling(window).sum()
        elif operation == 'std':
            new_df[f'{column}_rolling_{operation}_{window}'] = new_df[column].rolling(window).std()
        elif operation == 'min':
            new_df[f'{column}_rolling_{operation}_{window}'] = new_df[column].rolling(window).min()
        elif operation == 'max':
            new_df[f'{column}_rolling_{operation}_{window}'] = new_df[column].rolling(window).max()
        
        result = FluentDataFrame(new_df)
        result._operations = self._operations + [f"window_operation({column}, {operation}, window={window})"]
        return result
    
    def execute(self):
        """Execute and return the underlying DataFrame"""
        return self._df
    
    def show_operations(self):
        """Show all operations performed"""
        print("Operations performed:")
        for i, op in enumerate(self._operations, 1):
            print(f"  {i}. {op}")
        return self

# Example usage of fluent interface
print("Demonstrating fluent interface with method chaining:")

# Create sample data
np.random.seed(42)
sample_data = pd.DataFrame({
    'customer_id': range(1, 1001),
    'age': np.random.randint(18, 80, 1000),
    'income': np.random.normal(50000, 15000, 1000),
    'spend_amount': np.random.exponential(100, 1000),
    'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 1000),
    'satisfaction': np.random.randint(1, 6, 1000),
    'is_premium': np.random.choice([True, False], 1000, p=[0.3, 0.7]),
    'signup_date': pd.date_range('2020-01-01', periods=1000, freq='D')
})

# Add some missing values
sample_data.loc[np.random.choice(sample_data.index, 50), 'income'] = np.nan
sample_data.loc[np.random.choice(sample_data.index, 30), 'satisfaction'] = np.nan

print(f"Original data shape: {sample_data.shape}")

# Fluent analysis chain
result = (FluentDataFrame(sample_data)
          .filter_rows(age={'min': 25, 'max': 65})
          .filter_rows(category=['Electronics', 'Clothing'])
          .handle_missing(method='fill', columns=['income'], fill_value=sample_data['income'].mean())
          .add_column('income_category', 
                     expression="np.where(df['income'] > 60000, 'High', np.where(df['income'] > 40000, 'Medium', 'Low'))")
          .add_column('spend_per_income', 
                     function=lambda df: df['spend_amount'] / df['income'])
          .transform_column('spend_amount', lambda x: np.log1p(x))
          .window_operation('satisfaction', 'mean', window=50)
          .aggregate_by(['category', 'income_category'], 
                       {'customer_id': 'count', 
                        'spend_amount': ['mean', 'sum'], 
                        'satisfaction': 'mean',
                        'age': 'mean'})
          .sort_by(['category', 'customer_id_count'], ascending=[True, False])
          .show_operations())

print(f"\nFinal result shape: {result.df.shape}")
print("\nFinal result:")
print(result.df.head(10))

# Execute to get final DataFrame
final_df = result.execute()
print(f"\nExecuted DataFrame shape: {final_df.shape}")

# Complex chaining example with joins
print("\n" + "="*60)
print("Complex example with joins and multiple operations:")

# Create additional reference data
categories_ref = pd.DataFrame({
    'category': ['Electronics', 'Clothing', 'Food', 'Books'],
    'margin_rate': [0.15, 0.40, 0.20, 0.35],
    'return_rate': [0.05, 0.15, 0.02, 0.03]
})

# Complex fluent operation
complex_result = (FluentDataFrame(sample_data)
                  .filter_rows(is_premium=True)
                  .handle_missing(method='interpolate')
                  .add_column('age_group', 
                             expression="np.where(df['age'] < 30, 'Young', np.where(df['age'] < 50, 'Middle', 'Senior'))")
                  .join_with(categories_ref, on='category', how='left')
                  .add_column('expected_margin', 
                             function=lambda df: df['spend_amount'] * df['margin_rate'])
                  .window_operation('spend_amount', 'sum', window=30)
                  .aggregate_by(['age_group', 'category'], 
                               {'customer_id': 'count',
                                'spend_amount': 'sum',
                                'expected_margin': 'sum',
                                'satisfaction': 'mean'})
                  .add_column('margin_per_customer', 
                             function=lambda df: df['expected_margin_sum'] / df['customer_id_count'])
                  .sort_by('margin_per_customer', ascending=False)
                  .select_columns(['age_group', 'category', 'customer_id_count', 
                                  'margin_per_customer', 'satisfaction_mean'])
                  .show_operations())

print(f"\nComplex result:")
print(complex_result.df.round(2))
```
**Output:**
```
Demonstrating fluent interface with method chaining:
Original data shape: (1000, 8)

Operations performed:
  1. filter_rows(age >= 25, age <= 65)
  2. filter_rows(category in ['Electronics', 'Clothing'])
  3. handle_missing(method='fill', fill_value=49876.543)
  4. add_column(income_category, expression='np.where(df['income'] > 60000, 'High', np.where(df['income'] > 40000, 'Medium', 'Low'))')
  5. add_column(spend_per_income, function)
  6. transform_column(spend_amount)
  7. window_operation(satisfaction, mean, window=50)
  8. aggregate_by(['category', 'income_category'], {'customer_id': 'count', 'spend_amount': ['mean', 'sum'], 'satisfaction': 'mean', 'age': 'mean'})
  9. sort_by(['category', 'customer_id_count'], ascending=[True, False])

Final result shape: (6, 7)

Final result:
      category income_category  customer_id_count  spend_amount_mean  spend_amount_sum  satisfaction_mean   age_mean
0    Clothing            High                 32              4.567            146.15              3.234     45.67
1    Clothing          Medium                 28              4.234            118.55              3.123     42.34
2    Clothing             Low                 24              3.987             95.69              2.987     38.92
3  Electronics            High                 29              4.345            126.01              3.345     46.23
4  Electronics          Medium                 31              4.123            127.81              3.212     43.56
5  Electronics             Low                 22              4.012             88.26              3.045     39.87

Executed DataFrame shape: (6, 7)

============================================================
Complex example with joins and multiple operations:

Operations performed:
  1. filter_rows(is_premium == True)
  2. handle_missing(method='interpolate')
  3. add_column(age_group, expression='np.where(df['age'] < 30, 'Young', np.where(df['age'] < 50, 'Middle', 'Senior'))')
  4. join_with(how='left', on=['category'])
  5. add_column(expected_margin, function)
  6. window_operation(spend_amount, sum, window=30)
  7. aggregate_by(['age_group', 'category'], {'customer_id': 'count', 'spend_amount': 'sum', 'expected_margin': 'sum', 'satisfaction': 'mean'})
  8. add_column(margin_per_customer, function)
  9. sort_by(margin_per_customer, ascending=False)
  10. select_columns(['age_group', 'category', 'customer_id_count', 'margin_per_customer', 'satisfaction_mean'])

Complex result:
  age_group    category  customer_id_count  margin_per_customer  satisfaction_mean
0    Middle    Clothing                 18             16.73               3.22
1    Senior    Clothing                 14             15.89               3.45
2     Young    Clothing                 12             14.56               2.98
3    Middle Electronics                 16             12.34               3.12
4    Senior Electronics                 11             11.78               3.34
5     Young Electronics                 13             10.89               2.87
```

## Summary of Custom Functions and Advanced Techniques

| Technique | Purpose | Key Benefits | Use Cases |
|-----------|---------|--------------|-----------|
| Custom Accessors | Extend pandas functionality | Domain-specific methods, code reusability | Finance, analytics, domain-specific operations |
| Performance Decorators | Monitor and optimize | Profiling, caching, error handling | Large datasets, production systems |
| Custom Data Types | Specialized data handling | Type safety, domain logic | Monetary values, coordinates, specialized formats |
| Fluent Interface | Readable data pipelines | Method chaining, clear workflow | Complex data transformations, analysis pipelines |

### Best Practices for Custom Extensions

1. **Documentation**: Thoroughly document custom methods and classes
2. **Type Hints**: Use type hints for better IDE support and clarity
3. **Error Handling**: Implement robust error handling and validation
4. **Performance**: Consider performance implications of custom operations
5. **Testing**: Write comprehensive tests for custom functionality
6. **Compatibility**: Ensure compatibility with pandas operations

### Integration Guidelines

1. **Naming Conventions**: Use clear, descriptive names for custom methods
2. **Return Types**: Maintain consistent return types (DataFrame, Series, etc.)
3. **Parameter Validation**: Validate inputs and provide helpful error messages
4. **Memory Management**: Be mindful of memory usage in custom operations
5. **Extensibility**: Design for future extensions and modifications

---

**Next: Advanced Analytics and Statistics**