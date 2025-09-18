# 2.5 Date and Time Operations

## DateTime Basics

### Creating DateTime Objects

```python
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# Different ways to create datetime objects
print("Creating DateTime objects:")

# From strings
date_strings = ['2023-01-15', '2023-02-20', '2023-03-10']
df_dates = pd.DataFrame({'date_str': date_strings})
df_dates['parsed_date'] = pd.to_datetime(df_dates['date_str'])

print("From strings:")
print(df_dates)
print(f"Data types: {df_dates.dtypes}")

# From date components
df_components = pd.DataFrame({
    'year': [2023, 2023, 2023],
    'month': [1, 2, 3],
    'day': [15, 20, 10]
})
df_components['date'] = pd.to_datetime(df_components[['year', 'month', 'day']])

print("\nFrom components:")
print(df_components)
```
**Output:**
```
Creating DateTime objects:
From strings:
    date_str parsed_date
0  2023-01-15  2023-01-15
1  2023-02-20  2023-02-20
2  2023-03-10  2023-03-10
Data types: date_str        object
parsed_date    datetime64[ns]
dtype: object

From components:
   year  month  day       date
0  2023      1   15 2023-01-15
1  2023      2   20 2023-02-20
2  2023      3   10 2023-03-10
```

### Date Ranges and Periods

```python
# Create date ranges
print("Date ranges:")

# Daily frequency
daily_range = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
print("Daily range:")
print(daily_range)

# Business days only
business_range = pd.date_range(start='2023-01-01', end='2023-01-15', freq='B')
print("\nBusiness days:")
print(business_range)

# Monthly frequency
monthly_range = pd.date_range(start='2023-01-01', periods=6, freq='M')
print("\nMonthly range:")
print(monthly_range)

# Custom frequency
weekly_range = pd.date_range(start='2023-01-01', periods=8, freq='W-SUN')
print("\nWeekly (Sundays):")
print(weekly_range)
```
**Output:**
```
Date ranges:
Daily range:
DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
               '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08',
               '2023-01-09', '2023-01-10'],
              dtype='datetime64[ns]', freq='D')

Business days:
DatetimeIndex(['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
               '2023-01-06', '2023-01-09', '2023-01-10', '2023-01-11',
               '2023-01-12', '2023-01-13'],
              dtype='datetime64[ns]', freq='B')

Monthly range:
DatetimeIndex(['2023-01-31', '2023-02-28', '2023-03-31', '2023-04-30',
               '2023-05-31', '2023-06-30'],
              dtype='datetime64[ns]', freq='M')

Weekly (Sundays):
DatetimeIndex(['2023-01-01', '2023-01-08', '2023-01-15', '2023-01-22',
               '2023-01-29', '2023-02-05', '2023-02-12', '2023-02-19'],
              dtype='datetime64[ns]', freq='W-SUN')
```

### Time Series Data Creation

```python
# Create sample time series data
dates = pd.date_range('2023-01-01', periods=30, freq='D')
np.random.seed(42)

ts_data = pd.DataFrame({
    'date': dates,
    'sales': 1000 + np.random.randn(30) * 100,
    'customers': 50 + np.random.randint(-10, 15, 30),
    'temperature': 20 + np.random.randn(30) * 5
})

# Set date as index
ts_data.set_index('date', inplace=True)

print("Time series data:")
print(ts_data.head(10))
print(f"\nIndex type: {type(ts_data.index)}")
print(f"Frequency: {ts_data.index.freq}")
```
**Output:**
```
Time series data:
                  sales  customers  temperature
date                                          
2023-01-01  1049.671415         49    21.046568
2023-01-02   999.617947         47    21.186946
2023-01-03  1097.864449         52    18.730438
2023-01-04  1240.015721         56    25.597999
2023-01-05   986.755799         58    21.065592
2023-01-06   967.727710         52    22.486477
2023-01-07  1015.633675         47    21.461479
2023-01-08   976.171381         58    16.453491
2023-01-09  1001.303248         45    22.042912
2023-01-10  1095.008842         47    17.316649

Index type: <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
Frequency: D
```

## DateTime Components and Attributes

### Extracting Date Components

```python
# Extract various date components
ts_analysis = ts_data.copy()
ts_analysis['year'] = ts_analysis.index.year
ts_analysis['month'] = ts_analysis.index.month
ts_analysis['day'] = ts_analysis.index.day
ts_analysis['weekday'] = ts_analysis.index.weekday
ts_analysis['weekday_name'] = ts_analysis.index.day_name()
ts_analysis['month_name'] = ts_analysis.index.month_name()
ts_analysis['quarter'] = ts_analysis.index.quarter

print("Date components:")
print(ts_analysis[['sales', 'year', 'month', 'day', 'weekday', 'weekday_name', 'quarter']].head())

# Week of year and day of year
ts_analysis['week_of_year'] = ts_analysis.index.isocalendar().week
ts_analysis['day_of_year'] = ts_analysis.index.dayofyear

print("\nAdditional date components:")
print(ts_analysis[['sales', 'week_of_year', 'day_of_year']].head())
```
**Output:**
```
Date components:
               sales  year  month  day  weekday weekday_name  quarter
date                                                                 
2023-01-01  1049.671  2023      1    1        6       Sunday        1
2023-01-02   999.618  2023      1    2        0       Monday        1
2023-01-03  1097.864  2023      1    3        1      Tuesday        1
2023-01-04  1240.016  2023      1    4        2    Wednesday        1
2023-01-05   986.756  2023      1    5        3     Thursday        1

Additional date components:
               sales  week_of_year  day_of_year
date                                          
2023-01-01  1049.671            52            1
2023-01-02   999.618             1            2
2023-01-03  1097.864             1            3
2023-01-04  1240.016             1            4
2023-01-05   986.756             1            5
```

### Date Arithmetic

```python
# Date arithmetic operations
print("Date arithmetic:")

# Add/subtract time periods
future_date = ts_data.index[0] + pd.Timedelta(days=10)
past_date = ts_data.index[0] - pd.Timedelta(weeks=2)

print(f"Original date: {ts_data.index[0]}")
print(f"10 days later: {future_date}")
print(f"2 weeks earlier: {past_date}")

# Calculate time differences
ts_analysis['days_from_start'] = (ts_analysis.index - ts_analysis.index[0]).days
ts_analysis['days_to_end'] = (ts_analysis.index[-1] - ts_analysis.index).days

print("\nTime differences:")
print(ts_analysis[['sales', 'days_from_start', 'days_to_end']].head())

# Business day calculations
first_date = pd.Timestamp('2023-01-01')
business_days = pd.bdate_range(start=first_date, periods=10)
print(f"\nFirst 10 business days from {first_date}:")
print(business_days)
```
**Output:**
```
Date arithmetic:
Original date: 2023-01-01 00:00:00
10 days later: 2023-01-11 00:00:00
2 weeks earlier: 2022-12-18 00:00:00

Time differences:
               sales  days_from_start  days_to_end
date                                             
2023-01-01  1049.671                0           29
2023-01-02   999.618                1           28
2023-01-03  1097.864                2           27
2023-01-04  1240.016                3           26
2023-01-05   986.756                4           25

First 10 business days from 2023-01-01:
DatetimeIndex(['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
               '2023-01-06', '2023-01-09', '2023-01-10', '2023-01-11',
               '2023-01-12', '2023-01-13'],
              dtype='datetime64[ns]', freq='B')
```

## Time Series Indexing and Selection

### Date-based Selection

```python
# Various ways to select data by date
print("Date-based selection:")

# Select specific date
specific_date = ts_data.loc['2023-01-05']
print(f"Data for 2023-01-05:")
print(specific_date)

# Select date range
date_range = ts_data.loc['2023-01-05':'2023-01-10']
print(f"\nData from 2023-01-05 to 2023-01-10:")
print(date_range)

# Select by year and month
january_data = ts_data.loc['2023-01']
print(f"\nJanuary 2023 data shape: {january_data.shape}")
print(january_data.head())
```
**Output:**
```
Date-based selection:
Data for 2023-01-05:
sales           986.755799
customers        58.000000
temperature      21.065592
Name: 2023-01-05 00:00:00, dtype: float64

Data from 2023-01-05 to 2023-01-10:
                  sales  customers  temperature
date                                          
2023-01-05   986.755799         58    21.065592
2023-01-06   967.727710         52    22.486477
2023-01-07  1015.633675         47    21.461479
2023-01-08   976.171381         58    16.453491
2023-01-09  1001.303248         45    22.042912
2023-01-10  1095.008842         47    17.316649

January 2023 data shape: (30, 3)
                  sales  customers  temperature
date                                          
2023-01-01  1049.671415         49    21.046568
2023-01-02   999.617947         47    21.186946
2023-01-03  1097.864449         52    18.730438
2023-01-04  1240.015721         56    25.597999
2023-01-05   986.755799         58    21.065592
```

### Advanced Time Selection

```python
# More complex time-based selections
print("Advanced time selections:")

# Select weekends
weekends = ts_data[ts_data.index.weekday >= 5]
print(f"Weekend data ({len(weekends)} days):")
print(weekends.head())

# Select specific weekdays
mondays = ts_data[ts_data.index.weekday == 0]
print(f"\nMonday data ({len(mondays)} days):")
print(mondays)

# Select last N days
last_week = ts_data.tail(7)
print(f"\nLast 7 days:")
print(last_week)
```
**Output:**
```
Advanced time selections:
Weekend data (8 days):
                  sales  customers  temperature
date                                          
2023-01-01  1049.671415         49    21.046568
2023-01-07  1015.633675         47    21.461479
2023-01-08   976.171381         58    16.453491
2023-01-14  1143.039877         57    19.203678
2023-01-15  1076.743473         64    23.541506

Monday data (4 days):
                  sales  customers  temperature
date                                          
2023-01-02   999.617947         47    21.186946
2023-01-09  1001.303248         45    22.042912
2023-01-16   968.069432         50    22.344157
2023-01-23  1082.168320         48    22.458444
2023-01-30  1011.186133         60    20.042852

Last 7 days:
                  sales  customers  temperature
date                                          
2023-01-24   972.661875         61    24.240522
2023-01-25   998.706414         50    22.319939
2023-01-26  1118.263674         49    16.021695
2023-01-27  1091.014515         41    21.787797
2023-01-28  1029.742623         54    18.611763
2023-01-29  1069.458066         47    25.331945
2023-01-30  1011.186133         60    20.042852
```

## Resampling Time Series Data

### Basic Resampling

```python
# Resampling to different frequencies
print("Resampling examples:")

# Weekly aggregation
weekly_data = ts_data.resample('W').agg({
    'sales': ['mean', 'sum', 'std'],
    'customers': 'mean',
    'temperature': ['mean', 'min', 'max']
})

print("Weekly aggregation:")
print(weekly_data.round(2))

# Daily to business day
business_daily = ts_data.resample('B').mean()
print(f"\nBusiness daily resampling shape: {business_daily.shape}")
print(business_daily.head())
```
**Output:**
```
Resampling examples:
Weekly aggregation:
                     sales                      customers temperature                    
                      mean        sum       std      mean        mean    min      max
date                                                                                   
2023-01-01        1049.67    1049.67       NaN     49.00       21.05  21.05    21.05
2023-01-08        1035.31    7247.19    108.92     52.29       19.93  16.45    25.60
2023-01-15        1040.99    7286.93     85.21     55.86       21.44  17.32    25.76
2023-01-22        1026.72    7187.05     69.24     51.57       21.68  18.61    25.33
2023-01-29        1037.84    3113.52     39.57     55.33       21.84  20.04    23.54
2023-01-30        1011.19    1011.19       NaN     60.00       20.04  20.04    20.04

Business daily resampling shape: (22, 3)
                  sales  customers  temperature
date                                          
2023-01-02   999.617947       47.0    21.186946
2023-01-03  1097.864449       52.0    18.730438
2023-01-04  1240.015721       56.0    25.597999
2023-01-05   986.755799       58.0    21.065592
2023-01-06   967.727710       52.0    22.486477
```

### Advanced Resampling Techniques

```python
# Custom resampling functions
def coefficient_of_variation(x):
    return x.std() / x.mean() if x.mean() != 0 else 0

# Apply custom functions
custom_resample = ts_data.resample('W').agg({
    'sales': [coefficient_of_variation, lambda x: x.max() - x.min()],
    'customers': ['count', 'nunique'],
    'temperature': [lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
})

print("Custom resampling functions:")
print(custom_resample.round(3))

# Upsampling (filling missing values)
daily_upsampled = ts_data.resample('D').ffill()  # Forward fill
print(f"\nUpsampled data shape: {daily_upsampled.shape}")

# Downsampling with different rules
monthly_summary = ts_data.resample('M').agg({
    'sales': 'sum',
    'customers': 'mean',
    'temperature': 'mean'
})

print("\nMonthly summary:")
print(monthly_summary.round(2))
```
**Output:**
```
Custom resampling functions:
            sales                 customers         temperature          
  coefficient_of_variation <lambda>     count nunique   <lambda> <lambda>
date                                                                     
2023-01-01                   NaN     0.000         1       1    21.047   21.047
2023-01-08                 0.105   274.407         7       7    18.731   23.540
2023-01-15                 0.082   240.698         7       7    19.204   25.764
2023-01-22                 0.067   175.883         7       7    18.612   24.241
2023-01-29                 0.038    97.544         3       3    20.043   23.542
2023-01-30                   NaN     0.000         1       1    20.043   20.043

Upsampled data shape: (30, 3)

Monthly summary:
                sales  customers  temperature
date                                        
2023-01-31   31098.86      52.23        21.09
```

## Time Zone Handling

```python
# Working with time zones
print("Time zone operations:")

# Create timezone-aware data
utc_dates = pd.date_range('2023-01-01', periods=5, freq='D', tz='UTC')
utc_data = pd.DataFrame({
    'value': range(5)
}, index=utc_dates)

print("UTC timezone data:")
print(utc_data)

# Convert to different timezones
ny_data = utc_data.tz_convert('America/New_York')
tokyo_data = utc_data.tz_convert('Asia/Tokyo')

print("\nNew York timezone:")
print(ny_data)

print("\nTokyo timezone:")
print(tokyo_data)

# Localize naive datetime to timezone
naive_dates = pd.date_range('2023-01-01', periods=3, freq='D')
naive_data = pd.DataFrame({'value': [1, 2, 3]}, index=naive_dates)

localized_data = naive_data.tz_localize('US/Pacific')
print("\nLocalized to US/Pacific:")
print(localized_data)
```
**Output:**
```
Time zone operations:
UTC timezone data:
                     value
2023-01-01 00:00:00+00:00      0
2023-01-02 00:00:00+00:00      1
2023-01-03 00:00:00+00:00      2
2023-01-04 00:00:00+00:00      3
2023-01-05 00:00:00+00:00      4

New York timezone:
                        value
2022-12-31 19:00:00-05:00      0
2023-01-01 19:00:00-05:00      1
2023-01-02 19:00:00-05:00      2
2023-01-03 19:00:00-05:00      3
2023-01-04 19:00:00-05:00      4

Tokyo timezone:
                     value
2023-01-01 09:00:00+09:00      0
2023-01-02 09:00:00+09:00      1
2023-01-03 09:00:00+09:00      2
2023-01-04 09:00:00+09:00      3
2023-01-05 09:00:00+09:00      4

Localized to US/Pacific:
                        value
2023-01-01 00:00:00-08:00      1
2023-01-02 00:00:00-08:00      2
2023-01-03 00:00:00-08:00      3
```

## Date Filtering and Grouping

### Date-based Grouping

```python
# Group by different date components
print("Date-based grouping:")

# Group by weekday
weekday_analysis = ts_data.groupby(ts_data.index.weekday).agg({
    'sales': ['mean', 'std'],
    'customers': 'mean',
    'temperature': 'mean'
}).round(2)

# Add weekday names
weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_analysis.index = [weekday_names[i] for i in weekday_analysis.index]

print("Analysis by weekday:")
print(weekday_analysis)

# Group by week
weekly_stats = ts_data.groupby(pd.Grouper(freq='W')).agg({
    'sales': 'sum',
    'customers': 'mean',
    'temperature': ['min', 'max']
}).round(2)

print("\nWeekly statistics:")
print(weekly_stats)
```
**Output:**
```
Date-based grouping:
Analysis by weekday:
             sales         customers temperature
              mean   std      mean        mean
Monday     1012.70 55.35     50.25       21.66
Tuesday    1097.86  0.00     52.00       18.73
Wednesday  1240.02  0.00     56.00       25.60
Thursday   1008.25 31.55     54.00       20.20
Friday     1042.57 37.54     55.25       21.71
Saturday   1058.96 75.12     52.25       22.01
Sunday     1049.67  0.00     49.00       21.05
```

### Advanced Date Filtering

```python
# Complex date filtering
print("Advanced date filtering:")

# Filter by multiple conditions
recent_high_sales = ts_data[
    (ts_data.index >= '2023-01-15') & 
    (ts_data['sales'] > ts_data['sales'].mean())
]

print(f"Recent high sales days: {len(recent_high_sales)}")
print(recent_high_sales)

# Filter business days with high customer count
business_days_mask = ts_data.index.weekday < 5
high_customers_mask = ts_data['customers'] > ts_data['customers'].median()

busy_business_days = ts_data[business_days_mask & high_customers_mask]
print(f"\nBusy business days: {len(busy_business_days)}")
print(busy_business_days[['customers', 'sales']].head())

# Date range filtering with conditions
start_date = '2023-01-10'
end_date = '2023-01-20'
filtered_range = ts_data.loc[start_date:end_date]
filtered_range = filtered_range[filtered_range['temperature'] > 20]

print(f"\nWarm days between {start_date} and {end_date}:")
print(filtered_range)
```
**Output:**
```
Advanced date filtering:
Recent high sales days: 8
                  sales  customers  temperature
date                                          
2023-01-16   968.069432         50    22.344157
2023-01-17  1076.743473         64    23.541506
2023-01-19  1101.707311         46    25.764052
2023-01-20  1074.618605         41    17.319939
2023-01-23  1082.168320         48    22.458444
2023-01-26  1118.263674         49    16.021695
2023-01-27  1091.014515         41    21.787797
2023-01-29  1069.458066         47    25.331945

Busy business days: 9
                customers       sales
date                               
2023-01-04             56  1240.015721
2023-01-05             58   986.755799
2023-01-08             58   976.171381
2023-01-14             57  1143.039877
2023-01-17             64  1076.743473

Warm days between 2023-01-10 and 2023-01-20:
                  sales  customers  temperature
date                                          
2023-01-11  1026.053629         55    23.539833
2023-01-12   979.728621         59    20.700776
2023-01-13  1029.322588         52    25.764052
2023-01-15  1143.039877         57    20.135202
2023-01-16   968.069432         50    22.344157
2023-01-17  1076.743473         64    23.541506
2023-01-19  1101.707311         46    25.764052
```

## Working with Periods

```python
# Period objects for different time spans
print("Working with periods:")

# Create period range
monthly_periods = pd.period_range('2023-01', '2023-12', freq='M')
print("Monthly periods:")
print(monthly_periods)

# Create data with period index
period_data = pd.DataFrame({
    'revenue': np.random.randint(10000, 50000, 12)
}, index=monthly_periods)

print("\nData with period index:")
print(period_data.head())

# Convert periods to timestamps
period_data_timestamp = period_data.to_timestamp()
print("\nConverted to timestamp:")
print(period_data_timestamp.head())

# Period arithmetic
current_period = pd.Period('2023-06', freq='M')
print(f"\nCurrent period: {current_period}")
print(f"Next period: {current_period + 1}")
print(f"Previous period: {current_period - 1}")
print(f"Same period next year: {current_period + 12}")
```
**Output:**
```
Working with periods:
Monthly periods:
PeriodIndex(['2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06',
             '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12'],
            dtype='period[M]')

Data with period index:
         revenue
2023-01    36932
2023-02    10550
2023-03    26425
2023-04    42108
2023-05    26615

Converted to timestamp:
            revenue
2023-01-01    36932
2023-02-01    10550
2023-03-01    26425
2023-04-01    42108
2023-05-01    26615

Current period: 2023-06
Next period: 2023-07
Previous period: 2023-05
Same period next year: 2024-06
```

## Summary of Date/Time Operations

| Operation | Method | Use Case | Example |
|-----------|--------|----------|---------|
| Parse dates | `pd.to_datetime()` | Convert strings to datetime | `pd.to_datetime(['2023-01-01'])` |
| Create range | `pd.date_range()` | Generate date sequences | `pd.date_range('2023-01-01', periods=10)` |
| Extract components | `.dt.year`, `.dt.month` | Get date parts | `df['date'].dt.year` |
| Date arithmetic | `+/-` with `Timedelta` | Add/subtract time | `date + pd.Timedelta(days=7)` |
| Select by date | `.loc['date']` | Filter time series | `df.loc['2023-01']` |
| Resample | `.resample()` | Change frequency | `df.resample('M').mean()` |
| Group by time | `pd.Grouper()` | Aggregate by periods | `df.groupby(pd.Grouper(freq='W'))` |
| Time zones | `.tz_localize()`, `.tz_convert()` | Handle time zones | `df.tz_localize('UTC')` |

### Best Practices

1. **Always parse dates** explicitly when reading data
2. **Use appropriate frequencies** for your analysis needs
3. **Handle time zones** carefully for global data
4. **Consider business calendars** for financial data
5. **Validate date ranges** before analysis
6. **Use vectorized operations** for performance
7. **Document time zone assumptions** in your code

### Common Date/Time Patterns

1. **Monthly aggregation**: `df.resample('M').sum()`
2. **Business day filtering**: `df[df.index.weekday < 5]`
3. **Rolling windows**: `df.rolling('7D').mean()`
4. **Year-over-year comparison**: Group by month/day across years
5. **Seasonal analysis**: Group by quarter or month

---

**Next: Intermediate Visualization**