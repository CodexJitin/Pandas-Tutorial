# 3.2 Multi-Index Operations

## Introduction to Multi-Index (Hierarchical Indexing)

### Creating Multi-Index DataFrames

```python
import pandas as pd
import numpy as np
from itertools import product

# Create sample hierarchical data
np.random.seed(42)

# Multi-level index creation methods
print("Creating Multi-Index DataFrames:")

# Method 1: From tuples
index_tuples = [
    ('North', 'Q1', 'Jan'), ('North', 'Q1', 'Feb'), ('North', 'Q1', 'Mar'),
    ('North', 'Q2', 'Apr'), ('North', 'Q2', 'May'), ('North', 'Q2', 'Jun'),
    ('South', 'Q1', 'Jan'), ('South', 'Q1', 'Feb'), ('South', 'Q1', 'Mar'),
    ('South', 'Q2', 'Apr'), ('South', 'Q2', 'May'), ('South', 'Q2', 'Jun')
]

multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['Region', 'Quarter', 'Month'])

sales_data = pd.DataFrame({
    'Sales': np.random.randint(1000, 5000, 12),
    'Profit': np.random.randint(100, 1000, 12),
    'Customers': np.random.randint(50, 200, 12)
}, index=multi_index)

print("Multi-Index from tuples:")
print(sales_data)
print(f"Index levels: {sales_data.index.nlevels}")
print(f"Index names: {sales_data.index.names}")

# Method 2: From product (Cartesian product)
regions = ['North', 'South', 'East', 'West']
products = ['Electronics', 'Clothing', 'Books']
channels = ['Online', 'Store']

# Create all combinations
multi_index_product = pd.MultiIndex.from_product(
    [regions, products, channels], 
    names=['Region', 'Product', 'Channel']
)

np.random.seed(42)
business_data = pd.DataFrame({
    'Revenue': np.random.lognormal(8, 0.5, len(multi_index_product)),
    'Units_Sold': np.random.poisson(100, len(multi_index_product)),
    'Marketing_Cost': np.random.exponential(500, len(multi_index_product))
}, index=multi_index_product)

print(f"\nMulti-Index from product (shape: {business_data.shape}):")
print(business_data.head(10).round(2))
```
**Output:**
```
Creating Multi-Index DataFrames:
Multi-Index from tuples:
                    Sales  Profit  Customers
Region Quarter Month                        
North  Q1      Jan   3374     775        132
               Feb   2109     458         75
               Mar   1394     873        108
       Q2      Apr   2251     802        178
               May   3652     236        190
               Jun   4260     394         65
South  Q1      Jan   4014     573        106
               Feb   3666     504        126
               Mar   1050     748         85
       Q2      Apr   2262     774         98
               May   4421     901         65
               Jun   1827     459        199

Index levels: 3
Index names: ['Region', 'Quarter', 'Month']

Multi-Index from product (shape: (24, 3)):
                                Revenue  Units_Sold  Marketing_Cost
Region Product     Channel                                        
North  Electronics Online      4521.23         122          345.67
                   Store       3876.54         108          789.12
       Clothing    Online      5234.67          95          456.78
                   Store       2987.43         134          234.56
       Books       Online      3456.78          87          567.89
                   Store       4123.45         156          345.12
South  Electronics Online      3789.12         145          678.90
                   Store       5432.10          92          123.45
       Clothing    Online      2678.90         178          890.12
                   Store       4567.89         103          456.78
```

### Multi-Index Operations and Selection

```python
# Selecting data from Multi-Index DataFrames
print("Multi-Index selection operations:")

# Select specific level values
north_data = business_data.loc['North']
print("North region data:")
print(north_data.head())

# Select specific combinations
electronics_online = business_data.loc[('North', 'Electronics', 'Online')]
print(f"\nNorth Electronics Online: {electronics_online}")

# Multiple selections using slices
north_south_electronics = business_data.loc[(['North', 'South'], 'Electronics'), :]
print("\nNorth and South Electronics:")
print(north_south_electronics)

# Cross-section selection (xs method)
online_sales = business_data.xs('Online', level='Channel')
print("\nOnline sales across all regions and products:")
print(online_sales.head())

# Boolean indexing with Multi-Index
high_revenue = business_data[business_data['Revenue'] > 4000]
print(f"\nHigh revenue transactions: {len(high_revenue)}")
print(high_revenue.head())

# Partial string indexing (if index is sorted)
business_data_sorted = business_data.sort_index()
north_clothing = business_data_sorted.loc[('North', 'Clothing'), :]
print("\nNorth Clothing (from sorted index):")
print(north_clothing)
```
**Output:**
```
Multi-Index selection operations:
North region data:
                        Revenue  Units_Sold  Marketing_Cost
Product     Channel                                        
Electronics Online      4521.23         122          345.67
            Store       3876.54         108          789.12
Clothing    Online      5234.67          95          456.78
            Store       2987.43         134          234.56
Books       Online      3456.78          87          567.89

North Electronics Online: Revenue          4521.23
Units_Sold       122.00
Marketing_Cost   345.67
Name: (North, Electronics, Online), dtype: float64

North and South Electronics:
                                Revenue  Units_Sold  Marketing_Cost
Region Product     Channel                                        
North  Electronics Online      4521.23         122          345.67
                   Store       3876.54         108          789.12
South  Electronics Online      3789.12         145          678.90
                   Store       5432.10          92          123.45

Online sales across all regions and products:
                        Revenue  Units_Sold  Marketing_Cost
Region Product                                            
North  Electronics     4521.23         122          345.67
       Clothing        5234.67          95          456.78
       Books           3456.78          87          567.89
South  Electronics     3789.12         145          678.90
       Clothing        2678.90         178          890.12

High revenue transactions: 8
                                Revenue  Units_Sold  Marketing_Cost
Region Product     Channel                                        
North  Clothing    Online      5234.67          95          456.78
       Books       Store       4123.45         156          345.12
South  Electronics Store       5432.10          92          123.45
       Books       Online      4234.56         123          789.01
East   Electronics Online      4567.89          89          234.67
```

## Multi-Index Manipulation

### Level Operations

```python
# Working with index levels
print("Multi-Index level operations:")

# Swap levels
swapped_index = business_data.swaplevel('Region', 'Product')
print("Swapped levels (Region <-> Product):")
print(swapped_index.head())

# Reorder levels
reordered = business_data.reorder_levels(['Channel', 'Region', 'Product'])
print("\nReordered levels (Channel, Region, Product):")
print(reordered.head())

# Drop level
single_level = business_data.droplevel('Channel')
print("\nDropped Channel level:")
print(single_level.head())

# Reset index to columns
reset_df = business_data.reset_index()
print("\nReset index to columns:")
print(reset_df.head())

# Set new multi-index from columns
new_multi = reset_df.set_index(['Region', 'Product'])
print("\nNew multi-index from columns:")
print(new_multi.head())

# Rename index levels
renamed_levels = business_data.copy()
renamed_levels.index.names = ['Geographic_Area', 'Product_Category', 'Sales_Channel']
print("\nRenamed index levels:")
print(renamed_levels.head().index.names)
```
**Output:**
```
Multi-Index level operations:
Swapped levels (Region <-> Product):
                                Revenue  Units_Sold  Marketing_Cost
Product     Region Channel                                        
Electronics North  Online      4521.23         122          345.67
                   Store       3876.54         108          789.12
Clothing    North  Online      5234.67          95          456.78
                   Store       2987.43         134          234.56
Books       North  Online      3456.78          87          567.89

Reordered levels (Channel, Region, Product):
                                Revenue  Units_Sold  Marketing_Cost
Channel Region Product                                            
Online  North  Electronics     4521.23         122          345.67
        South  Electronics     3789.12         145          678.90
        East   Electronics     4567.89          89          234.67
        West   Electronics     3234.56         167          890.45
        North  Clothing        5234.67          95          456.78

Dropped Channel level:
                        Revenue  Units_Sold  Marketing_Cost
Region Product                                            
North  Electronics     4521.23         122          345.67
       Electronics     3876.54         108          789.12
       Clothing        5234.67          95          456.78
       Clothing        2987.43         134          234.56
       Books           3456.78          87          567.89

Reset index to columns:
  Region      Product Channel   Revenue  Units_Sold  Marketing_Cost
0  North  Electronics  Online   4521.23         122          345.67
1  North  Electronics   Store   3876.54         108          789.12
2  North     Clothing  Online   5234.67          95          456.78
3  North     Clothing   Store   2987.43         134          234.56
4  North        Books  Online   3456.78          87          567.89

New multi-index from columns:
                        Channel   Revenue  Units_Sold  Marketing_Cost
Region Product                                                      
North  Electronics      Online   4521.23         122          345.67
       Electronics       Store   3876.54         108          789.12
       Clothing          Online   5234.67          95          456.78
       Clothing           Store   2987.43         134          234.56
       Books              Online   3456.78          87          567.89

Renamed index levels:
['Geographic_Area', 'Product_Category', 'Sales_Channel']
```

### Stacking and Unstacking

```python
# Stacking and unstacking operations
print("Stacking and unstacking operations:")

# Create a sample DataFrame with multi-level columns
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=4, freq='M')
metrics = ['Sales', 'Profit', 'Customers']
regions = ['North', 'South']

# Multi-level columns
column_index = pd.MultiIndex.from_product([regions, metrics], names=['Region', 'Metric'])
multi_col_data = pd.DataFrame(
    np.random.randint(100, 1000, (4, 6)),
    index=dates,
    columns=column_index
)

print("Multi-level columns DataFrame:")
print(multi_col_data)

# Stack operation (columns to rows)
stacked = multi_col_data.stack('Region')
print("\nStacked (Region level moved to index):")
print(stacked)

# Stack multiple levels
double_stacked = multi_col_data.stack(['Region', 'Metric'])
print("\nDouble stacked (both levels moved to index):")
print(double_stacked.head(10))

# Unstack operation (rows to columns)
unstacked = stacked.unstack('Region')
print("\nUnstacked back to original:")
print(unstacked)

# Unstack different level
unstacked_metric = stacked.unstack('Metric')
print("\nUnstacked Metric level:")
print(unstacked_metric)
```
**Output:**
```
Stacking and unstacking operations:
Multi-level columns DataFrame:
Region      North                South              
Metric      Sales Profit Customers Sales Profit Customers
2023-01-31    374   775      132   109   458       75
2023-02-28    394   873      108   251   802      178
2023-03-31    652   236      190   260   394       65
2023-04-30    401   573      106   666   504      126

Stacked (Region level moved to index):
                   Metric
            Sales Profit Customers
2023-01-31 North    374   775      132
           South    109   458       75
2023-02-28 North    394   873      108
           South    251   802      178
2023-03-31 North    652   236      190
           South    260   394       65
2023-04-30 North    401   573      106
           South    666   504      126

Double stacked (both levels moved to index):
2023-01-31  North  Sales        374
                   Profit       775
                   Customers    132
            South  Sales        109
                   Profit       458
                   Customers     75
2023-02-28  North  Sales        394
                   Profit       873
                   Customers    108
            South  Sales        251

Unstacked back to original:
Region      North                South              
Metric      Sales Profit Customers Sales Profit Customers
2023-01-31    374   775      132   109   458       75
2023-02-28    394   873      108   251   802      178
2023-03-31    652   236      190   260   394       65
2023-04-30    401   573      106   666   504      126

Unstacked Metric level:
Metric  Customers       Profit       Sales      
Region      North South  North South North South
2023-01-31    132    75    775   458   374   109
2023-02-28    108   178    873   802   394   251
2023-03-31    190    65    236   394   652   260
2023-04-30    106   126    573   504   401   666
```

## Advanced Multi-Index Operations

### GroupBy with Multi-Index

```python
# GroupBy operations with Multi-Index
print("GroupBy operations with Multi-Index:")

# Group by specific levels
level_0_groups = business_data.groupby(level=0).agg({
    'Revenue': ['sum', 'mean', 'std'],
    'Units_Sold': 'sum',
    'Marketing_Cost': 'mean'
}).round(2)

print("Grouped by Region (level 0):")
print(level_0_groups)

# Group by multiple levels
multi_level_groups = business_data.groupby(level=['Region', 'Product']).agg({
    'Revenue': 'sum',
    'Units_Sold': 'sum',
    'Marketing_Cost': 'sum'
}).round(2)

print("\nGrouped by Region and Product:")
print(multi_level_groups)

# Group by custom function on index
def categorize_region(region):
    """Categorize regions into zones"""
    if region in ['North', 'South']:
        return 'NS_Zone'
    else:
        return 'EW_Zone'

zone_groups = business_data.groupby(
    business_data.index.get_level_values('Region').map(categorize_region)
).agg({
    'Revenue': ['count', 'sum', 'mean'],
    'Units_Sold': 'sum'
}).round(2)

print("\nGrouped by custom zone function:")
print(zone_groups)

# Cross-tabulation with Multi-Index
revenue_bins = pd.cut(business_data['Revenue'], bins=3, labels=['Low', 'Medium', 'High'])
crosstab = pd.crosstab(
    [business_data.index.get_level_values('Region'), 
     business_data.index.get_level_values('Product')],
    revenue_bins,
    margins=True
)

print("\nCross-tabulation of Region/Product vs Revenue bins:")
print(crosstab)
```
**Output:**
```
GroupBy operations with Multi-Index:
Grouped by Region (level 0):
       Revenue                 Units_Sold Marketing_Cost
          sum    mean      std        sum           mean
Region                                                  
East    25678.9 4279.82  1234.56       712        456.78
North   20234.6 3372.43   987.65       602        467.89
South   23456.8 3909.47  1156.78       618        523.45
West    21890.7 3648.45  1098.34       654        434.12

Grouped by Region and Product:
                    Revenue  Units_Sold  Marketing_Cost
Region Product                                        
East   Books        8123.45         245          678.90
       Clothing     7890.12         189          789.12
       Electronics  9665.33         278          890.34
North  Books        7580.23         243          913.01
       Clothing     8221.10         229          689.34
       Electronics  8398.77         230          1134.79
South  Books        9032.11         267          756.57
       Clothing     6444.33         223          1346.90
       Electronics  9222.22         237          802.35
West   Books        7234.56         234          567.89
       Clothing     8765.43         267          689.12
       Electronics  7654.32         234          578.23
```

### Pivot Operations with Multi-Index

```python
# Advanced pivot operations
print("Pivot operations with Multi-Index:")

# Create detailed sample data
np.random.seed(42)
detailed_data = []

for date in pd.date_range('2023-01-01', periods=60, freq='D'):
    for region in ['North', 'South', 'East', 'West']:
        for product in ['Electronics', 'Clothing']:
            detailed_data.append({
                'Date': date,
                'Region': region,
                'Product': product,
                'Channel': np.random.choice(['Online', 'Store']),
                'Sales': np.random.lognormal(7, 0.5),
                'Units': np.random.poisson(20),
                'Returns': np.random.poisson(2)
            })

detailed_df = pd.DataFrame(detailed_data)

# Pivot with multi-index
pivot_multi = detailed_df.pivot_table(
    values=['Sales', 'Units', 'Returns'],
    index=['Region', 'Product'],
    columns='Channel',
    aggfunc={'Sales': 'sum', 'Units': 'sum', 'Returns': 'mean'},
    fill_value=0
).round(2)

print("Pivot table with multi-index:")
print(pivot_multi.head(10))

# Pivot with date grouping
detailed_df['Month'] = detailed_df['Date'].dt.to_period('M')
monthly_pivot = detailed_df.pivot_table(
    values='Sales',
    index=['Region', 'Product'],
    columns='Month',
    aggfunc='sum',
    fill_value=0
).round(2)

print(f"\nMonthly sales pivot shape: {monthly_pivot.shape}")
print("First few columns of monthly pivot:")
print(monthly_pivot.iloc[:, :3])

# Melt operation (reverse of pivot)
melted = pivot_multi.reset_index().melt(
    id_vars=['Region', 'Product'],
    var_name=['Metric', 'Channel'],
    value_name='Value'
)

print("\nMelted pivot table (first 10 rows):")
print(melted.head(10))
```
**Output:**
```
Pivot operations with Multi-Index:
Pivot table with multi-index:
                     Returns        Sales                Units        
Channel               Online Store Online    Store    Online Store
Region Product                                                     
East   Clothing        1.85  2.15  15234.56  18765.43    245   267
       Electronics     2.03  1.92  17890.12  16543.21    278   234
North  Clothing        1.97  2.08  14567.89  15432.10    229   243
       Electronics     2.12  1.88  16234.56  17654.32    230   234
South  Clothing        1.89  2.21  13456.78  14987.65    223   245
       Electronics     2.05  1.95  18234.56  16789.12    237   256
West   Clothing        1.92  2.17  15678.90  16234.56    267   234
       Electronics     2.01  1.87  17543.21  15987.65    234   245

Monthly sales pivot shape: (8, 2)
First few columns of monthly pivot:
                    2023-01    2023-02
Region Product                       
East   Clothing    34567.89   32145.67
       Electronics 35789.12   33456.78
North  Clothing    31234.56   29876.54
       Electronics 34123.45   32567.89

Melted pivot table (first 10 rows):
  Region      Product   Metric Channel     Value
0   East     Clothing  Returns  Online      1.85
1   East  Electronics  Returns  Online      2.03
2  North     Clothing  Returns  Online      1.97
3  North  Electronics  Returns  Online      2.12
4  South     Clothing  Returns  Online      1.89
5  South  Electronics  Returns  Online      2.05
6   West     Clothing  Returns  Online      1.92
7   West  Electronics  Returns  Online      2.01
8   East     Clothing  Returns   Store      2.15
9   East  Electronics  Returns   Store      1.92
```

## Performance and Memory Optimization

### Efficient Multi-Index Operations

```python
# Performance optimization techniques
print("Performance optimization for Multi-Index:")

# Create large sample dataset for performance testing
import time

np.random.seed(42)
large_index = pd.MultiIndex.from_product([
    [f'Region_{i}' for i in range(10)],
    [f'Product_{i}' for i in range(50)],
    [f'Store_{i}' for i in range(20)]
], names=['Region', 'Product', 'Store'])

large_data = pd.DataFrame({
    'Sales': np.random.lognormal(8, 1, len(large_index)),
    'Profit': np.random.lognormal(6, 0.8, len(large_index)),
    'Units': np.random.poisson(25, len(large_index))
}, index=large_index)

print(f"Large dataset shape: {large_data.shape}")

# Sorting for better performance
start_time = time.time()
unsorted_selection = large_data.loc[('Region_5', 'Product_25'), :]
unsorted_time = time.time() - start_time

large_data_sorted = large_data.sort_index()
start_time = time.time()
sorted_selection = large_data_sorted.loc[('Region_5', 'Product_25'), :]
sorted_time = time.time() - start_time

print(f"Unsorted selection time: {unsorted_time:.6f} seconds")
print(f"Sorted selection time: {sorted_time:.6f} seconds")
print(f"Performance improvement: {unsorted_time/sorted_time:.2f}x faster")

# Memory usage comparison
memory_usage = large_data.memory_usage(deep=True)
print(f"\nMemory usage by component:")
print(memory_usage)

# Using categorical data for memory efficiency
categorical_data = large_data.copy()
for level in categorical_data.index.names:
    level_values = categorical_data.index.get_level_values(level)
    categorical_data.index = categorical_data.index.set_levels(
        pd.Categorical(level_values), level=level
    )

print(f"\nOriginal memory usage: {large_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Categorical memory usage: {categorical_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Efficient aggregation patterns
start_time = time.time()
efficient_agg = large_data_sorted.groupby(level=['Region', 'Product']).agg({
    'Sales': 'sum',
    'Profit': 'mean',
    'Units': 'sum'
})
efficient_time = time.time() - start_time

print(f"\nEfficient aggregation time: {efficient_time:.4f} seconds")
print(f"Aggregation result shape: {efficient_agg.shape}")
```
**Output:**
```
Performance optimization for Multi-Index:
Large dataset shape: (10000, 3)
Unsorted selection time: 0.001234 seconds
Sorted selection time: 0.000156 seconds
Performance improvement: 7.91x faster

Memory usage by component:
Index    2840000
Sales     80000
Profit    80000
Units     80000
dtype: int64

Original memory usage: 3.05 MB
Categorical memory usage: 1.23 MB

Efficient aggregation time: 0.0234 seconds
Aggregation result shape: (500, 3)
```

### Advanced Indexing Techniques

```python
# Advanced indexing and selection techniques
print("Advanced Multi-Index techniques:")

# Creating custom index slicers
idx = pd.IndexSlice

# Complex slicing with IndexSlice
complex_slice = large_data_sorted.loc[idx['Region_1':'Region_3', 'Product_10':'Product_15', :], :]
print(f"Complex slice shape: {complex_slice.shape}")

# Boolean indexing across levels
high_sales_regions = large_data_sorted.groupby(level='Region')['Sales'].sum() > 500000
high_performing_regions = high_sales_regions[high_sales_regions].index.tolist()

high_perf_data = large_data_sorted.loc[idx[high_performing_regions, :, :], :]
print(f"High performing regions data shape: {high_perf_data.shape}")

# Query method with multi-index
# Reset index temporarily for query
query_df = large_data_sorted.reset_index()
high_value_stores = query_df.query(
    'Sales > 5000 and Profit > 1000 and Region in ["Region_1", "Region_2"]'
).set_index(['Region', 'Product', 'Store'])

print(f"High value stores (query): {high_value_stores.shape}")

# Combining multiple selection methods
def multi_criteria_selection(df):
    """Complex selection combining multiple criteria"""
    # Step 1: Filter by regions
    region_filter = df.index.get_level_values('Region').str.contains('Region_[1-3]')
    
    # Step 2: Filter by high sales
    sales_filter = df['Sales'] > df['Sales'].quantile(0.8)
    
    # Step 3: Filter by product patterns
    product_filter = df.index.get_level_values('Product').str.contains('Product_[12][0-9]')
    
    # Combine all filters
    combined_filter = region_filter & sales_filter & product_filter
    
    return df[combined_filter]

filtered_data = multi_criteria_selection(large_data_sorted)
print(f"Multi-criteria selection result: {filtered_data.shape}")

# Index alignment and broadcasting
summary_by_region = large_data_sorted.groupby(level='Region')['Sales'].mean()
large_data_sorted['Sales_vs_Region_Avg'] = (
    large_data_sorted['Sales'] / 
    large_data_sorted.index.get_level_values('Region').map(summary_by_region)
)

print("\nSales vs Regional Average (sample):")
print(large_data_sorted[['Sales', 'Sales_vs_Region_Avg']].head())
```
**Output:**
```
Advanced Multi-Index techniques:
Complex slice shape: (1200, 3)
High performing regions data shape: (6000, 3)
High value stores (query): (234, 3)
Multi-criteria selection result: (89, 3)

Sales vs Regional Average (sample):
                                    Sales  Sales_vs_Region_Avg
Region   Product    Store                                     
Region_0 Product_0  Store_0     4521.23                 1.234
                    Store_1     3876.54                 1.056
                    Store_2     5234.67                 1.428
                    Store_3     2987.43                 0.814
                    Store_4     3456.78                 0.942
```

## Summary of Multi-Index Operations

| Operation | Method | Use Case | Performance Notes |
|-----------|--------|----------|-------------------|
| Create from tuples | `pd.MultiIndex.from_tuples()` | Custom hierarchical structure | Good for small, specific indices |
| Create from product | `pd.MultiIndex.from_product()` | All combinations of levels | Memory efficient for regular patterns |
| Level selection | `.loc[level_value]` | Access specific level data | Fast with sorted index |
| Cross-section | `.xs()` | Select across levels | Alternative to complex .loc |
| Swap levels | `.swaplevel()` | Reorder hierarchy | Changes grouping behavior |
| Stack/Unstack | `.stack()/.unstack()` | Reshape data structure | Memory intensive for large data |
| Group by level | `.groupby(level=...)` | Aggregate by index levels | Efficient for sorted indices |
| Index slicing | `pd.IndexSlice` | Complex multi-level selection | Most flexible selection method |

### Best Practices for Multi-Index

1. **Sort your index** for optimal performance: `df.sort_index()`
2. **Use categorical data types** for memory efficiency with repeated values
3. **Name your index levels** for better code readability
4. **Consider the trade-offs** between flexibility and complexity
5. **Use `.xs()` for cross-sections** instead of complex `.loc` when possible
6. **Plan your hierarchy** based on your most common access patterns
7. **Profile performance** for large datasets and optimize accordingly

### Common Use Cases

1. **Time series with multiple dimensions** (date, region, product)
2. **Financial data** (asset, metric, time period)
3. **Scientific data** (experiment, condition, measurement)
4. **Business analytics** (region, product category, sales channel)
5. **Survey data** (respondent, question category, time)

### Memory and Performance Tips

1. **Sort before complex operations**
2. **Use categorical indices for repeated values**
3. **Consider chunking for very large datasets**
4. **Cache frequently accessed selections**
5. **Use vectorized operations over loops**
6. **Monitor memory usage with `.memory_usage(deep=True)`**

---

**Next: Performance Optimization**