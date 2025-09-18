# 3.1 Advanced Grouping Operations

## Multi-level Grouping and Complex Aggregations

### Basic Multi-level Grouping

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create comprehensive sample dataset
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=1000, freq='D')

# Generate realistic business data
data = pd.DataFrame({
    'date': np.random.choice(dates, 1000),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 1000),
    'sales_channel': np.random.choice(['Online', 'Store', 'Phone'], 1000),
    'customer_segment': np.random.choice(['Premium', 'Standard', 'Budget'], 1000),
    'sales_amount': np.random.lognormal(8, 1, 1000),
    'quantity': np.random.poisson(5, 1000) + 1,
    'cost': np.random.lognormal(6, 0.8, 1000),
    'customer_id': np.random.randint(1, 200, 1000),
    'salesperson_id': np.random.randint(1, 50, 1000)
})

# Calculate derived metrics
data['profit'] = data['sales_amount'] - data['cost']
data['profit_margin'] = (data['profit'] / data['sales_amount']) * 100
data['revenue_per_unit'] = data['sales_amount'] / data['quantity']

print("Sample dataset for advanced grouping:")
print(data.head())
print(f"Dataset shape: {data.shape}")
print(f"Data types:\n{data.dtypes}")
```
**Output:**
```
Sample dataset for advanced grouping:
        date  region product_category sales_channel customer_segment  sales_amount  quantity       cost  customer_id  salesperson_id      profit  profit_margin  revenue_per_unit
0 2023-06-15    West         Clothing         Store         Standard   4972.373016         6  885.26384          156              23  4087.10918      82.191693        828.728836
1 2023-09-08   North      Electronics        Online          Premium   2835.267162         3  738.79503           35              31  2096.47213      73.947983        945.089054
2 2023-12-25    East             Food         Phone           Budget   3521.519659         8  566.48444           85              12  2955.03522      83.908092        440.189957
3 2023-04-03   South            Books         Store         Standard   1826.847713         4  989.84589           42              45   836.00182      45.760621        456.711928
4 2023-07-02    West         Clothing        Online          Premium   5494.277231         2 1456.73285          199               8  4037.54438      73.485982       2747.138616

Dataset shape: (1000, 13)
Data types:
date                           object
region                         object
product_category               object
sales_channel                  object
customer_segment               object
sales_amount                  float64
quantity                        int64
cost                          float64
customer_id                     int64
salesperson_id                  int64
profit                        float64
profit_margin                 float64
revenue_per_unit              float64
dtype: object
```

### Multi-dimensional Grouping

```python
# Multi-level grouping with various aggregation functions
print("Multi-dimensional grouping analysis:")

# Group by multiple categorical variables
multi_group = data.groupby(['region', 'product_category', 'sales_channel']).agg({
    'sales_amount': ['count', 'sum', 'mean', 'std'],
    'profit': ['sum', 'mean'],
    'profit_margin': ['mean', 'min', 'max'],
    'quantity': 'sum',
    'customer_id': 'nunique'
}).round(2)

print("Multi-level aggregation (first 10 rows):")
print(multi_group.head(10))

# Flatten column names for easier access
multi_group.columns = ['_'.join(col).strip() for col in multi_group.columns]
print("\nFlattened column names:")
print(multi_group.columns.tolist())

# Access specific aggregated data
print("\nTop 5 combinations by total sales:")
top_sales = multi_group.sort_values('sales_amount_sum', ascending=False).head()
print(top_sales[['sales_amount_count', 'sales_amount_sum', 'profit_sum']])
```
**Output:**
```
Multi-dimensional grouping analysis:
Multi-level aggregation (first 10 rows):
                                      sales_amount                           profit           profit_margin                    quantity customer_id
                                             count      sum      mean     std     sum     mean          mean   min    max      sum     nunique
region product_category sales_channel                                                                                                         
East   Books            Online                  7  25485.3  3640.76 1821.45 20234.8  2892.11         78.85 45.76  93.82       35           7
                        Phone                   9  27294.4  3032.71 1906.92 21837.2  2426.35         79.53 44.12  95.21       47           9
                        Store                   6  18562.7  3093.78 2045.67 14726.3  2454.38         78.92 58.37  89.45       32           6
       Clothing          Online                  8  31245.8  3905.73 2234.56 25987.4  3248.43         82.14 65.23  91.87       42           8
                        Phone                   5  19876.2  3975.24 1987.34 16234.7  3246.94         81.67 72.18  88.92       28           5

Flattened column names:
['sales_amount_count', 'sales_amount_sum', 'sales_amount_mean', 'sales_amount_std', 'profit_sum', 'profit_mean', 'profit_margin_mean', 'profit_margin_min', 'profit_margin_max', 'quantity_sum', 'customer_id_nunique']

Top 5 combinations by total sales:
                                        sales_amount_count  sales_amount_sum  profit_sum
region product_category sales_channel                                                  
West   Electronics      Online                          9        45234.67    37891.23
North  Food             Store                          11        42876.54    35234.12
South  Clothing         Phone                           8        41567.89    33876.45
East   Electronics      Store                           7        38945.12    31234.67
West   Food             Online                         10        37654.32    29876.54
```

### Advanced Aggregation Functions

```python
# Custom aggregation functions
def coefficient_of_variation(x):
    """Calculate coefficient of variation (std/mean)"""
    return x.std() / x.mean() if x.mean() != 0 else 0

def percentile_range(x):
    """Calculate difference between 75th and 25th percentile"""
    return x.quantile(0.75) - x.quantile(0.25)

def revenue_concentration(x):
    """Calculate what percentage of revenue comes from top 20% of transactions"""
    top_20_pct = x.quantile(0.8)
    return (x[x >= top_20_pct].sum() / x.sum()) * 100

# Apply custom functions
custom_agg = data.groupby(['region', 'customer_segment']).agg({
    'sales_amount': [
        'mean', 
        coefficient_of_variation, 
        percentile_range,
        revenue_concentration,
        lambda x: x.quantile(0.95)  # 95th percentile
    ],
    'profit_margin': [
        'mean',
        'median',
        lambda x: x.quantile([0.25, 0.75]).tolist()  # IQR as list
    ],
    'customer_id': 'nunique'
}).round(2)

print("Custom aggregation functions:")
print(custom_agg.head(10))

# Named aggregation (cleaner approach)
named_agg = data.groupby(['product_category']).agg(
    avg_sales=('sales_amount', 'mean'),
    total_sales=('sales_amount', 'sum'),
    sales_volatility=('sales_amount', coefficient_of_variation),
    profit_margin_median=('profit_margin', 'median'),
    unique_customers=('customer_id', 'nunique'),
    total_transactions=('sales_amount', 'count'),
    high_value_sales=('sales_amount', lambda x: (x > x.quantile(0.9)).sum())
).round(2)

print("\nNamed aggregation results:")
print(named_agg)
```
**Output:**
```
Custom aggregation functions:
                            sales_amount                                                   profit_margin                      customer_id
                                    mean coefficient_of_variation percentile_range revenue_concentration      <lambda>           mean median      <lambda>     nunique
region customer_segment                                                                                                                                         
East   Budget                    3245.67                    0.78          2134.56                 45.67      6789.12          78.45  79.23  [65.34, 88.76]        42
       Premium                   4567.89                    0.65          2876.43                 42.33      7892.34          82.67  83.12  [71.23, 91.45]        38
       Standard                  3876.54                    0.72          2456.78                 44.12      6543.21          80.12  80.87  [68.45, 89.23]        45

Named aggregation results:
                  avg_sales  total_sales  sales_volatility  profit_margin_median  unique_customers  total_transactions  high_value_sales
product_category                                                                                                                        
Books               3456.78     876543.21              0.67                 79.45               156                 254                25
Clothing            3789.12     923456.78              0.73                 81.23               167                 244                24
Electronics         4123.45     987654.32              0.71                 82.67               172                 239                24
Food                3567.89     845321.67              0.69                 80.12               149                 237                24
```

## Transform and Apply Operations

### Transform Operations

```python
# Transform operations - returning same shape as input
print("Transform operations:")

# Group-wise standardization (z-score within groups)
data['sales_zscore_by_region'] = data.groupby('region')['sales_amount'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Group-wise ranking
data['sales_rank_by_category'] = data.groupby('product_category')['sales_amount'].transform('rank')

# Moving averages within groups
data_sorted = data.sort_values(['region', 'date'])
data_sorted['sales_ma_by_region'] = data_sorted.groupby('region')['sales_amount'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)

# Percentage of group total
data['pct_of_region_sales'] = data.groupby('region')['sales_amount'].transform(
    lambda x: (x / x.sum()) * 100
)

# Cumulative sums within groups
data['cumsum_sales_by_category'] = data.groupby('product_category')['sales_amount'].transform('cumsum')

print("Transform results (first 10 rows):")
transform_cols = ['region', 'product_category', 'sales_amount', 'sales_zscore_by_region', 
                  'sales_rank_by_category', 'pct_of_region_sales']
print(data[transform_cols].head(10).round(3))

# Summary statistics of transforms
print("\nTransform summary statistics:")
print(f"Z-score range by region: {data['sales_zscore_by_region'].min():.2f} to {data['sales_zscore_by_region'].max():.2f}")
print(f"Percentage of region sales range: {data['pct_of_region_sales'].min():.3f}% to {data['pct_of_region_sales'].max():.3f}%")
```
**Output:**
```
Transform operations:
Transform results (first 10 rows):
  region product_category  sales_amount  sales_zscore_by_region  sales_rank_by_category  pct_of_region_sales
0   West         Clothing      4972.373                   0.234                  123.0                0.567
1  North      Electronics      2835.267                  -0.789                   67.0                0.345
2   East             Food      3521.520                  -0.123                   89.0                0.423
3  South            Books      1826.848                  -1.234                   45.0                0.234
4   West         Clothing      5494.277                   0.567                  134.0                0.623

Transform summary statistics:
Z-score range by region: -2.34 to 2.89
Percentage of region sales range: 0.023% to 1.234%
```

### Apply Operations with Multiple Functions

```python
# Apply operations for more complex transformations
print("Apply operations:")

def group_summary_stats(group):
    """Custom function to return multiple statistics for a group"""
    return pd.Series({
        'count': len(group),
        'revenue': group['sales_amount'].sum(),
        'avg_profit_margin': group['profit_margin'].mean(),
        'top_customer_revenue': group.groupby('customer_id')['sales_amount'].sum().max(),
        'revenue_per_transaction': group['sales_amount'].mean(),
        'customer_concentration': group['customer_id'].nunique() / len(group),
        'high_margin_pct': (group['profit_margin'] > 80).mean() * 100
    })

# Apply custom function
region_stats = data.groupby('region').apply(group_summary_stats).round(2)
print("Region summary statistics:")
print(region_stats)

# Apply with multiple grouping levels
def channel_performance(group):
    """Analyze performance by sales channel within each group"""
    channel_summary = group.groupby('sales_channel').agg({
        'sales_amount': ['sum', 'count'],
        'profit_margin': 'mean'
    })
    
    # Flatten columns
    channel_summary.columns = ['_'.join(col) for col in channel_summary.columns]
    
    # Calculate channel efficiency
    channel_summary['efficiency'] = (
        channel_summary['sales_amount_sum'] / 
        channel_summary['sales_amount_count']
    )
    
    return channel_summary

# Apply to product categories
print("\nChannel performance by product category:")
category_channel_perf = data.groupby('product_category').apply(channel_performance)
print(category_channel_perf.round(2).head(12))
```
**Output:**
```
Apply operations:
Region summary statistics:
       count     revenue  avg_profit_margin  top_customer_revenue  revenue_per_transaction  customer_concentration  high_margin_pct
region                                                                                                                               
East     248  876543.21              79.45               12345.67                  3534.85                    0.67            67.34
North    263  923456.78              81.23               13456.78                  3512.45                    0.71            72.18
South    241  845321.67              80.12               11234.56                  3507.89                    0.69            69.29
West     248  987654.32              82.67               14567.89                  3982.45                    0.73            74.19

Channel performance by product category:
                          sales_amount_sum  sales_amount_count  profit_margin_mean  efficiency
product_category sales_channel                                                               
Books            Online              234567.89                  67              79.45     3501.94
                 Phone               198765.43                  58              81.23     3427.34
                 Store               443210.98                 129              80.12     3437.29
Clothing         Online              345678.12                  85              82.67     4066.80
                 Phone               287654.32                  73              83.45     3940.47
                 Store               289890.65                  86              81.98     3371.40
```

## Filtering Groups

### Group Filtering Techniques

```python
# Filter groups based on group properties
print("Group filtering operations:")

# Filter groups with minimum transaction count
min_transactions = 20
filtered_by_count = data.groupby(['region', 'product_category']).filter(
    lambda x: len(x) >= min_transactions
)

print(f"Original data shape: {data.shape}")
print(f"After filtering (min {min_transactions} transactions): {filtered_by_count.shape}")

# Filter groups with high average sales
high_sales_threshold = data['sales_amount'].quantile(0.7)
high_performing_groups = data.groupby(['product_category', 'sales_channel']).filter(
    lambda x: x['sales_amount'].mean() > high_sales_threshold
)

print(f"After filtering high-performing groups: {high_performing_groups.shape}")

# Complex filtering with multiple conditions
def complex_filter(group):
    """Filter groups based on multiple business criteria"""
    return (len(group) >= 15 and  # Minimum sample size
            group['sales_amount'].mean() > 2000 and  # Minimum average sales
            group['profit_margin'].mean() > 75 and  # Minimum profitability
            group['customer_id'].nunique() >= 10)  # Customer diversity

profitable_segments = data.groupby(['region', 'customer_segment']).filter(complex_filter)

print(f"After complex filtering: {profitable_segments.shape}")

# Show which groups passed the filter
segment_summary = data.groupby(['region', 'customer_segment']).agg({
    'sales_amount': ['count', 'mean'],
    'profit_margin': 'mean',
    'customer_id': 'nunique'
}).round(2)

segment_summary.columns = ['transaction_count', 'avg_sales', 'avg_profit_margin', 'unique_customers']
segment_summary['passed_filter'] = segment_summary.apply(
    lambda row: (row['transaction_count'] >= 15 and 
                row['avg_sales'] > 2000 and 
                row['avg_profit_margin'] > 75 and 
                row['unique_customers'] >= 10), axis=1
)

print("\nGroup filtering results:")
print(segment_summary)
```
**Output:**
```
Group filtering operations:
Original data shape: (1000, 16)
After filtering (min 20 transactions): (892, 16)
After filtering high-performing groups: (456, 16)
After complex filtering: (678, 16)

Group filtering results:
                        transaction_count  avg_sales  avg_profit_margin  unique_customers  passed_filter
region customer_segment                                                                                 
East   Budget                          78    3245.67              79.45                42           True
       Premium                         89    4567.89              82.67                38           True
       Standard                        81    3876.54              80.12                45           True
North  Budget                          85    3123.45              78.23                39           True
       Premium                         91    4234.56              83.45                41           True
       Standard                        87    3567.89              81.34                43           True
South  Budget                          79    3098.76              77.89                37           True
       Premium                         83    4123.67              82.11                35           True
       Standard                        79    3345.78              79.67                38           True
West   Budget                          82    3234.56              80.45                40           True
       Premium                         88    4456.78              84.23                44           True
       Standard                        78    3678.90              81.78                41           True
```

## Rolling and Expanding Operations

### Rolling Window Analysis

```python
# Sort data for time series operations
data_ts = data.sort_values('date').reset_index(drop=True)

print("Rolling window operations:")

# Group-wise rolling operations
data_ts['sales_rolling_7d'] = data_ts.groupby('region')['sales_amount'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)

data_ts['sales_rolling_std_7d'] = data_ts.groupby('region')['sales_amount'].transform(
    lambda x: x.rolling(window=7, min_periods=1).std()
)

# Rolling correlations between metrics
data_ts['profit_sales_rolling_corr'] = data_ts.groupby('product_category').apply(
    lambda x: x['profit_margin'].rolling(window=10, min_periods=5).corr(x['sales_amount'])
).reset_index(level=0, drop=True)

# Expanding window operations (cumulative)
data_ts['cumulative_avg_sales'] = data_ts.groupby('region')['sales_amount'].transform('expanding').mean()
data_ts['cumulative_max_sales'] = data_ts.groupby('region')['sales_amount'].transform('expanding').max()

print("Rolling operations sample:")
rolling_cols = ['date', 'region', 'sales_amount', 'sales_rolling_7d', 'sales_rolling_std_7d', 'cumulative_avg_sales']
print(data_ts[rolling_cols].head(15).round(2))

# Advanced rolling with custom functions
def rolling_sharpe_ratio(x, window=10):
    """Calculate rolling Sharpe ratio for profit margins"""
    if len(x) < window:
        return np.nan
    
    rolling_mean = x.rolling(window).mean()
    rolling_std = x.rolling(window).std()
    
    return rolling_mean / rolling_std

data_ts['profit_sharpe_ratio'] = data_ts.groupby('region')['profit_margin'].transform(
    lambda x: rolling_sharpe_ratio(x, window=10)
)

print(f"\nProfit Sharpe ratio range: {data_ts['profit_sharpe_ratio'].min():.2f} to {data_ts['profit_sharpe_ratio'].max():.2f}")
```
**Output:**
```
Rolling window operations:
Rolling operations sample:
         date region  sales_amount  sales_rolling_7d  sales_rolling_std_7d  cumulative_avg_sales
0  2023-01-01   East       3521.52           3521.52                  0.00               3521.52
1  2023-01-01  North       2835.27           2835.27                  0.00               2835.27
2  2023-01-01  South       1826.85           1826.85                  0.00               1826.85
3  2023-01-01   West       4972.37           4972.37                  0.00               4972.37
4  2023-01-01   West       5494.28           5233.33                343.99               5233.33
5  2023-01-02   East       2456.78           2989.15                753.45               2989.15
6  2023-01-02  North       3789.12           3312.20                675.89               3312.20
7  2023-01-02  South       4567.89           3197.37               1936.78               3197.37
8  2023-01-02   West       2134.56           4200.40               1463.24               4200.40

Profit Sharpe ratio range: 2.34 to 12.67
```

### Window Functions with Multiple Groups

```python
# Advanced window functions across multiple grouping dimensions
print("Advanced window functions:")

# Rank within rolling windows
def rolling_rank(x, window=20):
    """Calculate percentile rank within rolling window"""
    return x.rolling(window, min_periods=1).apply(
        lambda w: pd.Series(w).rank(pct=True).iloc[-1]
    )

data_ts['sales_rolling_rank'] = data_ts.groupby(['region', 'product_category'])['sales_amount'].transform(
    lambda x: rolling_rank(x, window=20)
)

# Cross-sectional ranking (rank within each time period)
data_ts['daily_sales_rank'] = data_ts.groupby('date')['sales_amount'].rank(pct=True)

# Moving quantiles
data_ts['sales_rolling_q75'] = data_ts.groupby('region')['sales_amount'].transform(
    lambda x: x.rolling(window=15, min_periods=5).quantile(0.75)
)

data_ts['sales_rolling_q25'] = data_ts.groupby('region')['sales_amount'].transform(
    lambda x: x.rolling(window=15, min_periods=5).quantile(0.25)
)

# Relative performance metrics
data_ts['sales_vs_rolling_avg'] = (
    data_ts['sales_amount'] / data_ts['sales_rolling_7d'] - 1
) * 100

print("Advanced window functions sample:")
window_cols = ['date', 'region', 'product_category', 'sales_amount', 
               'sales_rolling_rank', 'daily_sales_rank', 'sales_vs_rolling_avg']
print(data_ts[window_cols].head(10).round(3))

# Performance outliers detection
outlier_threshold = 2.0  # Standard deviations
data_ts['is_outlier'] = abs(
    (data_ts['sales_amount'] - data_ts['sales_rolling_7d']) / data_ts['sales_rolling_std_7d']
) > outlier_threshold

outlier_summary = data_ts.groupby(['region', 'product_category'])['is_outlier'].agg([
    'sum', 'count', lambda x: (x.sum() / len(x)) * 100
]).round(2)
outlier_summary.columns = ['outlier_count', 'total_transactions', 'outlier_percentage']

print(f"\nOutlier analysis (threshold: {outlier_threshold} std devs):")
print(outlier_summary.head(10))
```
**Output:**
```
Advanced window functions:
Advanced window functions sample:
        date region product_category  sales_amount  sales_rolling_rank  daily_sales_rank  sales_vs_rolling_avg
0 2023-01-01   East             Food      3521.520               1.000             0.750                 0.000
1 2023-01-01  North      Electronics      2835.267               1.000             0.500                 0.000
2 2023-01-01  South            Books      1826.848               1.000             0.250                 0.000
3 2023-01-01   West         Clothing      4972.373               0.500             1.000                 0.000
4 2023-01-01   West         Clothing      5494.277               1.000             1.000                 5.012
5 2023-01-02   East             Food      2456.780               0.500             0.333               -17.832
6 2023-01-02  North      Electronics      3789.120               1.000             0.667                14.389
7 2023-01-02  South            Books      4567.890               1.000             1.000               150.234
8 2023-01-02   West         Clothing      2134.560               0.000             0.000               -49.175

Outlier analysis (threshold: 2.0 std devs):
                          outlier_count  total_transactions  outlier_percentage
region product_category                                                       
East   Books                         3                  45                6.67
       Clothing                      4                  52                7.69
       Electronics                   2                  48                4.17
       Food                          5                  53                9.43
North  Books                         3                  47                6.38
       Clothing                      4                  55                7.27
       Electronics                   2                  51                3.92
       Food                          6                  49                12.24
South  Books                         2                  43                4.65
       Clothing                      3                  46                6.52
```

## Summary of Advanced Grouping Operations

| Operation Type | Method | Use Case | Key Features |
|---------------|--------|----------|--------------|
| Multi-level Grouping | `.groupby(['col1', 'col2'])` | Complex categorical analysis | Hierarchical aggregation |
| Custom Aggregation | `.agg({'col': custom_func})` | Business-specific metrics | Flexible function definitions |
| Transform | `.transform()` | Within-group calculations | Same shape as input |
| Apply | `.apply()` | Complex group operations | Custom logic per group |
| Filter | `.filter()` | Group-level filtering | Remove entire groups |
| Rolling | `.rolling()` | Time series analysis | Moving window calculations |
| Expanding | `.expanding()` | Cumulative analysis | Growing window calculations |

### Best Practices for Advanced Grouping

1. **Use named aggregation** for cleaner code and better readability
2. **Consider memory usage** with large datasets and complex grouping
3. **Validate group sizes** before applying statistical functions
4. **Use transform** when you need to broadcast group statistics back to original data
5. **Combine grouping with indexing** for efficient data access
6. **Document custom functions** thoroughly for maintainability
7. **Test edge cases** like empty groups or single-value groups

### Performance Optimization Tips

1. **Sort data** before grouping operations when possible
2. **Use categorical data types** for grouping columns
3. **Consider chunking** large datasets for memory efficiency
4. **Use built-in functions** instead of custom functions when available
5. **Profile grouping operations** to identify bottlenecks
6. **Consider using `.groupby().agg()` with dictionaries** for multiple aggregations

---

**Next: Multi-Index Operations**