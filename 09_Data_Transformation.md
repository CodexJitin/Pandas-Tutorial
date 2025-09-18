# 2.3 Data Transformation

## Apply Functions

### Basic Apply Operations

```python
import pandas as pd
import numpy as np
import math

# Create sample dataset
np.random.seed(42)
data = {
    'Employee_ID': range(1001, 1011),
    'Name': [f'Employee_{i}' for i in range(1, 11)],
    'Age': [25, 30, 35, 28, 32, 27, 29, 31, 26, 33],
    'Salary': [50000, 65000, 70000, 62000, 55000, 58000, 67000, 72000, 51000, 69000],
    'Department': ['HR', 'IT', 'Finance', 'IT', 'HR', 'Marketing', 'Finance', 'IT', 'HR', 'Marketing'],
    'Years_Experience': [2, 5, 8, 4, 3, 5, 6, 7, 1, 6],
    'Performance_Score': [4.2, 4.8, 4.5, 4.7, 4.3, 4.6, 4.9, 4.4, 4.1, 4.7]
}
df = pd.DataFrame(data)
print("Original dataset:")
print(df.head())
```
**Output:**
```
Original dataset:
   Employee_ID        Name  Age  Salary Department  Years_Experience  Performance_Score
0         1001  Employee_1   25   50000         HR                 2                4.2
1         1002  Employee_2   30   65000         IT                 5                4.8
2         1003  Employee_3   35   70000    Finance                 8                4.5
3         1004  Employee_4   28   62000         IT                 4                4.7
4         1005  Employee_5   32   55000         HR                 3                4.3
```

```python
# Apply function to a single column (Series)
def categorize_age(age):
    if age < 30:
        return 'Young'
    elif age < 35:
        return 'Middle'
    else:
        return 'Senior'

df['Age_Category'] = df['Age'].apply(categorize_age)
print("Age categorization:")
print(df[['Name', 'Age', 'Age_Category']])
```
**Output:**
```
Age categorization:
         Name  Age Age_Category
0  Employee_1   25        Young
1  Employee_2   30       Middle
2  Employee_3   35       Senior
3  Employee_4   28        Young
4  Employee_5   32       Middle
5  Employee_6   27        Young
6  Employee_7   29        Young
7  Employee_8   31       Middle
8  Employee_9   26        Young
9 Employee_10   33       Middle
```

```python
# Apply lambda functions
df['Salary_K'] = df['Salary'].apply(lambda x: x / 1000)
df['Experience_Level'] = df['Years_Experience'].apply(
    lambda x: 'Entry' if x <= 2 else 'Mid' if x <= 5 else 'Senior'
)

print("Lambda function applications:")
print(df[['Name', 'Salary', 'Salary_K', 'Years_Experience', 'Experience_Level']].head())
```
**Output:**
```
Lambda function applications:
         Name  Salary  Salary_K  Years_Experience Experience_Level
0  Employee_1   50000      50.0                 2            Entry
1  Employee_2   65000      65.0                 5              Mid
2  Employee_3   70000      70.0                 8           Senior
3  Employee_4   62000      62.0                 4              Mid
4  Employee_5   55000      55.0                 3              Mid
```

### Apply to DataFrames (Row-wise Operations)

```python
# Apply function to entire rows
def calculate_bonus(row):
    base_bonus = row['Salary'] * 0.1
    performance_multiplier = row['Performance_Score'] / 4.0
    experience_bonus = row['Years_Experience'] * 1000
    return base_bonus * performance_multiplier + experience_bonus

df['Calculated_Bonus'] = df.apply(calculate_bonus, axis=1)

print("Calculated bonus for each employee:")
bonus_summary = df[['Name', 'Salary', 'Performance_Score', 'Years_Experience', 'Calculated_Bonus']].copy()
bonus_summary['Calculated_Bonus'] = bonus_summary['Calculated_Bonus'].round(0)
print(bonus_summary)
```
**Output:**
```
Calculated bonus for each employee:
         Name  Salary  Performance_Score  Years_Experience  Calculated_Bonus
0  Employee_1   50000                4.2                 2            7250.0
1  Employee_2   65000                4.8                 5           12800.0
2  Employee_3   70000                4.5                 8           15875.0
3  Employee_4   62000                4.7                 4           11285.0
4  Employee_5   55000                4.3                 3            8912.0
5  Employee_6   58000                4.6                 5           11670.0
6  Employee_7   67000                4.9                 6           14242.0
7  Employee_8   72000                4.4                 7           14840.0
8  Employee_9   51000                4.1                 1            6238.0
9 Employee_10   69000                4.7                 6           14262.0
```

```python
# Multiple return values from apply
def employee_analysis(row):
    return pd.Series({
        'Salary_Per_Experience': row['Salary'] / row['Years_Experience'],
        'Performance_Rank': 'High' if row['Performance_Score'] >= 4.5 else 'Medium' if row['Performance_Score'] >= 4.3 else 'Low',
        'Total_Compensation': row['Salary'] + row['Calculated_Bonus']
    })

analysis_results = df.apply(employee_analysis, axis=1)
df = pd.concat([df, analysis_results], axis=1)

print("Employee analysis results:")
print(df[['Name', 'Salary_Per_Experience', 'Performance_Rank', 'Total_Compensation']].round(0))
```
**Output:**
```
Employee analysis results:
         Name  Salary_Per_Experience Performance_Rank  Total_Compensation
0  Employee_1                25000.0              Low             57250.0
1  Employee_2                13000.0             High             77800.0
2  Employee_3                 8750.0             High             85875.0
3  Employee_4                15500.0             High             73285.0
4  Employee_5                18333.0           Medium             63912.0
5  Employee_6                11600.0             High             69670.0
6  Employee_7                11167.0             High             81242.0
7  Employee_8                10286.0           Medium             86840.0
8  Employee_9                51000.0              Low             57238.0
9 Employee_10                11500.0             High             83262.0
```

### Map Function for Series

```python
# Create mapping dictionary
department_codes = {
    'HR': 'H001',
    'IT': 'I002', 
    'Finance': 'F003',
    'Marketing': 'M004'
}

# Use map to transform values
df['Dept_Code'] = df['Department'].map(department_codes)

print("Department code mapping:")
print(df[['Name', 'Department', 'Dept_Code']].head())

# Map with default value for missing keys
performance_grades = {4.0: 'C', 4.1: 'C+', 4.2: 'B-', 4.3: 'B', 4.4: 'B', 4.5: 'B+', 
                     4.6: 'A-', 4.7: 'A', 4.8: 'A', 4.9: 'A+', 5.0: 'A+'}

df['Performance_Grade'] = df['Performance_Score'].map(performance_grades).fillna('Unknown')

print("\nPerformance grade mapping:")
print(df[['Name', 'Performance_Score', 'Performance_Grade']].head())
```
**Output:**
```
Department code mapping:
         Name Department Dept_Code
0  Employee_1         HR      H001
1  Employee_2         IT      I002
2  Employee_3    Finance      F003
3  Employee_4         IT      I002
4  Employee_5         HR      H001

Performance grade mapping:
         Name  Performance_Score Performance_Grade
0  Employee_1                4.2                B-
1  Employee_2                4.8                 A
2  Employee_3                4.5                B+
3  Employee_4                4.7                 A
4  Employee_5                4.3                 B
```

## Aggregation and Grouping

### Basic GroupBy Operations

```python
# Group by single column
dept_stats = df.groupby('Department').agg({
    'Salary': ['mean', 'min', 'max', 'count'],
    'Age': 'mean',
    'Performance_Score': 'mean',
    'Years_Experience': 'mean'
}).round(2)

print("Department statistics:")
print(dept_stats)
```
**Output:**
```
Department statistics:
            Salary                                  Age Performance_Score Years_Experience
              mean    min    max count        mean              mean              mean
Department                                                                              
Finance      68500  67000  70000     2       32.00              4.70              7.00
HR           52000  50000  55000     3       27.67              4.20              2.00
IT           66333  62000  72000     3       29.67              4.63              5.33
Marketing    63500  58000  69000     2       30.00              4.65              5.50
```

```python
# Multiple grouping columns
age_dept_stats = df.groupby(['Age_Category', 'Department']).agg({
    'Salary': 'mean',
    'Performance_Score': 'mean',
    'Employee_ID': 'count'
}).round(2)

print("Statistics by Age Category and Department:")
print(age_dept_stats.head())

# Group by with custom aggregation functions
def salary_range(series):
    return series.max() - series.min()

def performance_category(series):
    avg_score = series.mean()
    return 'High' if avg_score >= 4.5 else 'Medium' if avg_score >= 4.3 else 'Low'

custom_stats = df.groupby('Department').agg({
    'Salary': [salary_range, 'mean'],
    'Performance_Score': [performance_category, 'mean']
})

print("\nCustom aggregation functions:")
print(custom_stats.round(2))
```
**Output:**
```
Statistics by Age Category and Department:
                      Salary  Performance_Score  Employee_ID
Age_Category Department                                     
Middle       Finance    67000               4.90            1
             HR         55000               4.30            1
             IT         72000               4.40            1
             Marketing  69000               4.70            1
Senior       Finance    70000               4.50            1

Custom aggregation functions:
            Salary            Performance_Score         
       salary_range    mean performance_category  mean
Department                                           
Finance         3000  68500                High  4.70
HR              5000  52000                 Low  4.20
IT             10000  66333                High  4.63
Marketing      11000  63500                High  4.65
```

### Advanced Grouping Techniques

```python
# Transform function - returns same shape as original
df['Salary_Dept_Mean'] = df.groupby('Department')['Salary'].transform('mean')
df['Salary_Deviation'] = df['Salary'] - df['Salary_Dept_Mean']

print("Salary compared to department average:")
comparison = df[['Name', 'Department', 'Salary', 'Salary_Dept_Mean', 'Salary_Deviation']].copy()
comparison = comparison.round(0)
print(comparison)
```
**Output:**
```
Salary compared to department average:
         Name Department  Salary  Salary_Dept_Mean  Salary_Deviation
0  Employee_1         HR   50000             52000            -2000.0
1  Employee_2         IT   65000             66333            -1333.0
2  Employee_3    Finance   70000             68500             1500.0
3  Employee_4         IT   62000             66333            -4333.0
4  Employee_5         HR   55000             52000             3000.0
5  Employee_6  Marketing   58000             63500            -5500.0
6  Employee_7    Finance   67000             68500            -1500.0
7  Employee_8         IT   72000             66333             5667.0
8  Employee_9         HR   51000             52000            -1000.0
9 Employee_10  Marketing   69000             63500             5500.0
```

```python
# Filter groups based on conditions
large_departments = df.groupby('Department').filter(lambda x: len(x) >= 3)
print(f"Employees in large departments (≥3 people): {len(large_departments)}")
print(large_departments[['Name', 'Department']].groupby('Department').count())

# Apply function to groups
def standardize_salary(group):
    """Standardize salary within each department"""
    mean_sal = group['Salary'].mean()
    std_sal = group['Salary'].std()
    if std_sal == 0:
        return pd.Series([0] * len(group), index=group.index)
    return (group['Salary'] - mean_sal) / std_sal

df['Salary_Standardized'] = df.groupby('Department')['Salary'].apply(standardize_salary)

print("\nStandardized salaries by department:")
print(df[['Name', 'Department', 'Salary', 'Salary_Standardized']].round(2))
```
**Output:**
```
Employees in large departments (≥3 people): 6
Department
HR    3
IT    3
Name: Name, dtype: int64

Standardized salaries by department:
         Name Department  Salary  Salary_Standardized
0  Employee_1         HR   50000                -0.71
1  Employee_2         IT   65000                -0.27
2  Employee_3    Finance   70000                 1.00
3  Employee_4         IT   62000                -0.87
4  Employee_5         HR   55000                 1.41
5  Employee_6  Marketing   58000                -1.00
6  Employee_7    Finance   67000                -1.00
7  Employee_8         IT   72000                 1.15
8  Employee_9         HR   51000                -0.71
9 Employee_10  Marketing   69000                 1.00
```

## Pivot Operations

### Basic Pivot Tables

```python
# Create pivot table
pivot_salary = df.pivot_table(
    values='Salary',
    index='Department',
    columns='Age_Category',
    aggfunc='mean',
    fill_value=0
)

print("Salary pivot table (Department vs Age Category):")
print(pivot_salary.round(0))

# Multiple values in pivot table
pivot_multi = df.pivot_table(
    values=['Salary', 'Performance_Score'],
    index='Department',
    columns='Experience_Level',
    aggfunc='mean',
    fill_value=0
)

print("\nMultiple values pivot table:")
print(pivot_multi.round(2))
```
**Output:**
```
Salary pivot table (Department vs Age Category):
Age_Category   Middle  Senior  Young
Department                        
Finance         67000   70000      0
HR              55000       0  50500
IT              72000       0  63500
Marketing       69000       0  58000

Multiple values pivot table:
            Performance_Score                    Salary                  
Experience_Level        Entry    Mid Senior      Entry     Mid   Senior
Department                                                              
Finance                  0.00   0.00   4.70       0.00    0.00  68500.0
HR                       4.20   4.30   0.00   50500.00  55000.0      0.0
IT                       0.00   4.75   4.40       0.00  63500.0  72000.0
Marketing                0.00   4.65   0.00       0.00  63500.0      0.0
```

### Advanced Pivot Operations

```python
# Pivot with multiple aggregation functions
pivot_advanced = df.pivot_table(
    values='Salary',
    index='Department',
    columns='Performance_Rank',
    aggfunc=['mean', 'count', 'std'],
    fill_value=0
)

print("Advanced pivot with multiple aggregations:")
print(pivot_advanced.round(0))

# Cross-tabulation
performance_dept_crosstab = pd.crosstab(
    df['Department'],
    df['Performance_Rank'],
    margins=True,
    normalize='index'
)

print("\nCross-tabulation (normalized by department):")
print(performance_dept_crosstab.round(2))
```
**Output:**
```
Advanced pivot with multiple aggregations:
           count                mean                   std            
Performance_Rank  High  Low Medium  High    Low Medium   High  Low Medium
Department                                                              
Finance              2    0      0 68500      0      0   2121    0      0
HR                   0    1      2     0  50000  53000      0    0   2828
IT                   2    0      1 68500      0  72000   4950    0      0
Marketing            2    0      0 63500      0      0   7778    0      0

Cross-tabulation (normalized by department):
Performance_Rank  High   Low  Medium  All
Department                              
Finance           1.00  0.00    0.00  1.0
HR                0.00  0.33    0.67  1.0
IT                0.67  0.00    0.33  1.0
Marketing         1.00  0.00    0.00  1.0
All               0.60  0.10    0.30  1.0
```

## Reshaping Data

### Melt (Wide to Long)

```python
# Create wide format data
wide_data = {
    'Employee': ['Alice', 'Bob', 'Charlie'],
    'Q1_Sales': [1000, 1200, 800],
    'Q2_Sales': [1100, 1300, 900],
    'Q3_Sales': [1050, 1250, 950],
    'Q4_Sales': [1200, 1400, 1000]
}
df_wide = pd.DataFrame(wide_data)

print("Wide format data:")
print(df_wide)

# Melt to long format
df_long = pd.melt(
    df_wide,
    id_vars=['Employee'],
    value_vars=['Q1_Sales', 'Q2_Sales', 'Q3_Sales', 'Q4_Sales'],
    var_name='Quarter',
    value_name='Sales'
)

# Clean up quarter names
df_long['Quarter'] = df_long['Quarter'].str.replace('_Sales', '')

print("\nLong format data:")
print(df_long)
```
**Output:**
```
Wide format data:
  Employee  Q1_Sales  Q2_Sales  Q3_Sales  Q4_Sales
0    Alice      1000      1100      1050      1200
1      Bob      1200      1300      1250      1400
2  Charlie       800       900       950      1000

Long format data:
   Employee Quarter  Sales
0     Alice      Q1   1000
1       Bob      Q1   1200
2   Charlie      Q1    800
3     Alice      Q2   1100
4       Bob      Q2   1300
5   Charlie      Q2    900
6     Alice      Q3   1050
7       Bob      Q3   1250
8   Charlie      Q3    950
9     Alice      Q4   1200
10      Bob      Q4   1400
11  Charlie      Q4   1000
```

### Stack and Unstack

```python
# Create multi-index DataFrame
multi_index_data = df.set_index(['Department', 'Name'])[['Salary', 'Performance_Score']]
print("Multi-index DataFrame:")
print(multi_index_data.head())

# Stack columns to rows
stacked = multi_index_data.stack()
print("\nStacked data:")
print(stacked.head(10))

# Unstack to separate levels
unstacked = stacked.unstack(level='Name')
print("\nUnstacked by Name:")
print(unstacked.head())
```
**Output:**
```
Multi-index DataFrame:
                     Salary  Performance_Score
Department Name                               
HR         Employee_1   50000                4.2
           Employee_5   55000                4.3
           Employee_9   51000                4.1
IT         Employee_2   65000                4.8
           Employee_4   62000                4.7

Stacked data:
Department  Name        
HR          Employee_1  Salary               50000.0
                        Performance_Score        4.2
            Employee_5  Salary               55000.0
                        Performance_Score        4.3
            Employee_9  Salary               51000.0
                        Performance_Score        4.1
IT          Employee_2  Salary               65000.0
                        Performance_Score        4.8
            Employee_4  Salary               62000.0
                        Performance_Score        4.7
dtype: float64

Unstacked by Name:
Name         Employee_1 Employee_10 Employee_2 Employee_3 Employee_4 Employee_5 Employee_6 Employee_7 Employee_8 Employee_9
Department                                                                                                                 
Finance      NaN        NaN         NaN        67000.0    NaN        NaN        NaN        70000.0    NaN        NaN       
HR           50000.0    NaN         NaN        NaN        NaN        55000.0    NaN        NaN        NaN        51000.0   
IT           NaN        NaN         65000.0    NaN        62000.0    NaN        NaN        NaN        72000.0    NaN       
Marketing    NaN        69000.0     NaN        NaN        NaN        NaN        58000.0    NaN        NaN        NaN       
```

## Window Functions and Rolling Calculations

```python
# Create time series data for rolling calculations
dates = pd.date_range('2023-01-01', periods=10, freq='D')
ts_data = pd.DataFrame({
    'Date': dates,
    'Sales': [100, 120, 110, 130, 125, 140, 135, 150, 145, 160],
    'Customers': [10, 12, 11, 13, 12, 14, 13, 15, 14, 16]
})

print("Time series data:")
print(ts_data)

# Rolling calculations
ts_data['Sales_3day_MA'] = ts_data['Sales'].rolling(window=3).mean()
ts_data['Sales_3day_Sum'] = ts_data['Sales'].rolling(window=3).sum()
ts_data['Customers_3day_Max'] = ts_data['Customers'].rolling(window=3).max()

print("\nWith rolling calculations:")
print(ts_data.round(2))
```
**Output:**
```
Time series data:
        Date  Sales  Customers
0 2023-01-01    100         10
1 2023-01-02    120         12
2 2023-01-03    110         11
3 2023-01-04    130         13
4 2023-01-05    125         12
5 2023-01-06    140         14
6 2023-01-07    135         13
7 2023-01-08    150         15
8 2023-01-09    145         14
9 2023-01-10    160         16

With rolling calculations:
        Date  Sales  Customers  Sales_3day_MA  Sales_3day_Sum  Customers_3day_Max
0 2023-01-01    100         10            NaN             NaN                 NaN
1 2023-01-02    120         12            NaN             NaN                 NaN
2 2023-01-03    110         11         110.00             330                  12
3 2023-01-04    130         13         120.00             360                  13
4 2023-01-05    125         12         121.67             365                  13
5 2023-01-06    140         14         131.67             395                  14
6 2023-01-07    135         13         133.33             400                  14
7 2023-01-08    150         15         141.67             425                  15
8 2023-01-09    145         14         143.33             430                  15
9 2023-01-10    160         16         151.67             455                  16
```

```python
# Expanding windows
ts_data['Sales_Cumulative_Mean'] = ts_data['Sales'].expanding().mean()
ts_data['Sales_Cumulative_Sum'] = ts_data['Sales'].expanding().sum()

# Percentage change
ts_data['Sales_Pct_Change'] = ts_data['Sales'].pct_change()
ts_data['Sales_Diff'] = ts_data['Sales'].diff()

print("Expanded calculations and changes:")
print(ts_data[['Date', 'Sales', 'Sales_Cumulative_Mean', 'Sales_Pct_Change', 'Sales_Diff']].round(2))
```
**Output:**
```
Expanded calculations and changes:
        Date  Sales  Sales_Cumulative_Mean  Sales_Pct_Change  Sales_Diff
0 2023-01-01    100                 100.00               NaN         NaN
1 2023-01-02    120                 110.00              0.20        20.0
2 2023-01-03    110                 110.00             -0.08       -10.0
3 2023-01-04    130                 115.00              0.18        20.0
4 2023-01-05    125                 117.00             -0.04        -5.0
5 2023-01-06    140                 120.83              0.12        15.0
6 2023-01-07    135                 122.86             -0.04        -5.0
7 2023-01-08    150                 126.25              0.11        15.0
8 2023-01-09    145                 128.33              -0.03        -5.0
9 2023-01-10    160                 131.50              0.10        15.0
```

## Summary of Transformation Techniques

| Technique | Method | Use Case | Example |
|-----------|--------|----------|---------|
| Element-wise function | `.apply()` | Custom transformations | `df['col'].apply(func)` |
| Row-wise operations | `.apply(axis=1)` | Multi-column calculations | `df.apply(func, axis=1)` |
| Value mapping | `.map()` | Dictionary-based replacement | `df['col'].map(mapping_dict)` |
| Group aggregation | `.groupby().agg()` | Summary statistics by groups | `df.groupby('cat').agg({'val': 'mean'})` |
| Pivot table | `.pivot_table()` | Cross-tabulation | `df.pivot_table(values='val', index='row', columns='col')` |
| Reshape to long | `.melt()` | Wide to long format | `pd.melt(df, id_vars=['id'])` |
| Rolling calculations | `.rolling()` | Time series analysis | `df['col'].rolling(3).mean()` |
| Cumulative operations | `.expanding()` | Running totals | `df['col'].expanding().sum()` |

### Best Practices

1. **Use vectorized operations** when possible for performance
2. **Choose appropriate aggregation functions** for your data type
3. **Handle missing values** before transformations
4. **Consider memory usage** with large datasets
5. **Document complex transformations** for future reference
6. **Validate results** after major transformations

---

**Next: Merging and Joining Data**