# 1.6 Basic Data Manipulation

## Adding and Removing Columns

### Creating New Columns

```python
import pandas as pd
import numpy as np

# Create sample dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'Department': ['HR', 'IT', 'Finance', 'IT', 'HR'],
    'Salary': [50000, 65000, 70000, 62000, 55000],
    'Experience': [2, 5, 8, 4, 6]
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
```
**Output:**
```
Original DataFrame:
      Name  Age Department  Salary  Experience
0    Alice   25         HR   50000           2
1      Bob   30         IT   65000           5
2  Charlie   35    Finance   70000           8
3    Diana   28         IT   62000           4
4      Eve   32         HR   55000           6
```

```python
# Add new column with constant value
df['Country'] = 'USA'
print("After adding Country column:")
print(df)

# Add new column based on calculation
df['Annual_Bonus'] = df['Salary'] * 0.10
print("\nAfter adding Annual_Bonus column:")
print(df[['Name', 'Salary', 'Annual_Bonus']])
```
**Output:**
```
After adding Country column:
      Name  Age Department  Salary  Experience Country
0    Alice   25         HR   50000           2     USA
1      Bob   30         IT   65000           5     USA
2  Charlie   35    Finance   70000           8     USA
3    Diana   28         IT   62000           4     USA
4      Eve   32         HR   55000           6     USA

After adding Annual_Bonus column:
      Name  Salary  Annual_Bonus
0    Alice   50000        5000.0
1      Bob   65000        6500.0
2  Charlie   70000        7000.0
3    Diana   62000        6200.0
4      Eve   55000        5500.0
```

```python
# Add column based on conditional logic
df['Seniority'] = np.where(df['Experience'] >= 5, 'Senior', 'Junior')
print("After adding Seniority column:")
print(df[['Name', 'Experience', 'Seniority']])

# Add column using multiple conditions
conditions = [
    (df['Age'] < 30),
    (df['Age'] >= 30) & (df['Age'] < 35),
    (df['Age'] >= 35)
]
choices = ['Young', 'Middle', 'Senior']
df['Age_Group'] = np.select(conditions, choices, default='Unknown')
print("\nAfter adding Age_Group column:")
print(df[['Name', 'Age', 'Age_Group']])
```
**Output:**
```
After adding Seniority column:
      Name  Experience Seniority
0    Alice           2    Junior
1      Bob           5    Senior
2  Charlie           8    Senior
3    Diana           4    Junior
4      Eve           6    Senior

After adding Age_Group column:
      Name  Age Age_Group
0    Alice   25     Young
1      Bob   30    Middle
2  Charlie   35    Senior
3    Diana   28     Young
4      Eve   32    Middle
```

### Using assign() Method

```python
# Create multiple columns at once using assign()
df_assigned = df.assign(
    Salary_K = df['Salary'] / 1000,
    Total_Comp = df['Salary'] + df['Annual_Bonus'],
    Exp_Per_Age = df['Experience'] / df['Age']
)

print("New columns using assign():")
print(df_assigned[['Name', 'Salary_K', 'Total_Comp', 'Exp_Per_Age']].round(2))
```
**Output:**
```
New columns using assign():
      Name  Salary_K  Total_Comp  Exp_Per_Age
0    Alice      50.0     55000.0         0.08
1      Bob      65.0     71500.0         0.17
2  Charlie      70.0     77000.0         0.23
3    Diana      62.0     68200.0         0.14
4      Eve      55.0     60500.0         0.19
```

### Removing Columns

```python
# Remove columns using drop()
df_reduced = df.drop(['Country', 'Annual_Bonus'], axis=1)
print("After dropping columns:")
print(df_reduced.columns.tolist())

# Remove columns in place
df_copy = df.copy()
df_copy.drop(['Age_Group'], axis=1, inplace=True)
print("Columns after in-place drop:")
print(df_copy.columns.tolist())
```
**Output:**
```
After dropping columns:
['Name', 'Age', 'Department', 'Salary', 'Experience', 'Seniority']

Columns after in-place drop:
['Name', 'Age', 'Department', 'Salary', 'Experience', 'Country', 'Annual_Bonus', 'Seniority']
```

```python
# Remove columns using del
df_del = df.copy()
del df_del['Country']
print("After using del:")
print(df_del.columns.tolist())

# Remove multiple columns by selecting others
df_select = df[['Name', 'Age', 'Salary', 'Experience']]
print("After selecting specific columns:")
print(df_select.columns.tolist())
```
**Output:**
```
After using del:
['Name', 'Age', 'Department', 'Salary', 'Experience', 'Annual_Bonus', 'Seniority', 'Age_Group']

After selecting specific columns:
['Name', 'Age', 'Salary', 'Experience']
```

## Sorting Data

### Sorting by Values

```python
# Sort by single column
df_sorted = df.sort_values('Age')
print("Sorted by Age (ascending):")
print(df_sorted[['Name', 'Age', 'Salary']])

# Sort by single column (descending)
df_sorted_desc = df.sort_values('Salary', ascending=False)
print("\nSorted by Salary (descending):")
print(df_sorted_desc[['Name', 'Salary']])
```
**Output:**
```
Sorted by Age (ascending):
      Name  Age  Salary
0    Alice   25   50000
3    Diana   28   62000
1      Bob   30   65000
4      Eve   32   55000
2  Charlie   35   70000

Sorted by Salary (descending):
      Name  Salary
2  Charlie   70000
1      Bob   65000
3    Diana   62000
4      Eve   55000
0    Alice   50000
```

```python
# Sort by multiple columns
df_multi_sort = df.sort_values(['Department', 'Salary'], ascending=[True, False])
print("Sorted by Department (asc) then Salary (desc):")
print(df_multi_sort[['Name', 'Department', 'Salary']])

# Sort with missing values
df_with_na = df.copy()
df_with_na.loc[2, 'Age'] = np.nan
df_sorted_na = df_with_na.sort_values('Age', na_position='first')
print("\nSorted with NaN values first:")
print(df_sorted_na[['Name', 'Age']])
```
**Output:**
```
Sorted by Department (asc) then Salary (desc):
      Name Department  Salary
2  Charlie    Finance   70000
4      Eve         HR   55000
0    Alice         HR   50000
1      Bob         IT   65000
3    Diana         IT   62000

Sorted with NaN values first:
      Name   Age
2  Charlie   NaN
0    Alice  25.0
3    Diana  28.0
1      Bob  30.0
4      Eve  32.0
```

### Sorting by Index

```python
# Create DataFrame with custom index
df_indexed = df.set_index('Name')
print("DataFrame with Name as index:")
print(df_indexed.head())

# Sort by index
df_index_sorted = df_indexed.sort_index()
print("\nSorted by index (alphabetical):")
print(df_index_sorted.index.tolist())

# Sort index in descending order
df_index_desc = df_indexed.sort_index(ascending=False)
print("Index sorted descending:")
print(df_index_desc.index.tolist())
```
**Output:**
```
DataFrame with Name as index:
         Age Department  Salary  Experience Country  Annual_Bonus Seniority Age_Group
Name                                                                                  
Alice     25         HR   50000           2     USA        5000.0    Junior     Young
Bob       30         IT   65000           5     USA        6500.0    Senior    Middle
Charlie   35    Finance   70000           8     USA        7000.0    Senior    Senior
Diana     28         IT   62000           4     USA        6200.0    Junior     Young
Eve       32         HR   55000           6     USA        5500.0    Senior    Middle

Sorted by index (alphabetical):
['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']

Index sorted descending:
['Eve', 'Diana', 'Charlie', 'Bob', 'Alice']
```

## Basic Statistics

### Single Column Statistics

```python
# Basic statistics for a single column
salary_stats = df['Salary']
print("Salary statistics:")
print(f"Mean: ${salary_stats.mean():,.2f}")
print(f"Median: ${salary_stats.median():,.2f}")
print(f"Standard Deviation: ${salary_stats.std():,.2f}")
print(f"Minimum: ${salary_stats.min():,}")
print(f"Maximum: ${salary_stats.max():,}")
print(f"Sum: ${salary_stats.sum():,}")
print(f"Count: {salary_stats.count()}")
```
**Output:**
```
Salary statistics:
Mean: $60,400.00
Median: $62,000.00
Standard Deviation: $7,767.42
Minimum: 50,000
Maximum: 70,000
Sum: 302,000
Count: 5
```

```python
# Quantiles and percentiles
print("Salary percentiles:")
print(f"25th percentile: ${salary_stats.quantile(0.25):,.2f}")
print(f"50th percentile (median): ${salary_stats.quantile(0.5):,.2f}")
print(f"75th percentile: ${salary_stats.quantile(0.75):,.2f}")
print(f"90th percentile: ${salary_stats.quantile(0.9):,.2f}")

# Multiple quantiles at once
quantiles = salary_stats.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
print("\nMultiple quantiles:")
print(quantiles)
```
**Output:**
```
Salary percentiles:
25th percentile: $55,000.00
50th percentile (median): $62,000.00
75th percentile: $65,000.00
90th percentile: $68,000.00

Multiple quantiles:
0.10    52000.0
0.25    55000.0
0.50    62000.0
0.75    65000.0
0.90    68000.0
Name: Salary, dtype: float64
```

### DataFrame-wide Statistics

```python
# Statistics for all numeric columns
print("Statistics for all numeric columns:")
print(df.describe())

# Specific statistics for all numeric columns
print("\nMeans:")
print(df.mean(numeric_only=True))

print("\nStandard deviations:")
print(df.std(numeric_only=True))
```
**Output:**
```
Statistics for all numeric columns:
             Age        Salary    Experience  Annual_Bonus
count   5.000000      5.000000      5.000000      5.000000
mean   30.000000  60400.000000      5.000000   6040.000000
std     4.183300   7767.424642      2.449490    776.742464
min    25.000000  50000.000000      2.000000   5000.000000
25%    28.000000  55000.000000      4.000000   5500.000000
50%    30.000000  62000.000000      5.000000   6200.000000
75%    32.000000  65000.000000      6.000000   6500.000000
max    35.000000  70000.000000      8.000000   7000.000000

Means:
Age               30.0
Salary         60400.0
Experience         5.0
Annual_Bonus    6040.0
dtype: float64

Standard deviations:
Age              4.183300
Salary        7767.424642
Experience       2.449490
Annual_Bonus     776.742464
dtype: float64
```

### Aggregation by Groups

```python
# Group by department and calculate statistics
dept_stats = df.groupby('Department')['Salary'].agg(['mean', 'min', 'max', 'count'])
print("Salary statistics by Department:")
print(dept_stats)

# Multiple columns and functions
dept_multi_stats = df.groupby('Department').agg({
    'Salary': ['mean', 'sum'],
    'Age': 'mean',
    'Experience': ['min', 'max']
})
print("\nMultiple statistics by Department:")
print(dept_multi_stats.round(2))
```
**Output:**
```
Salary statistics by Department:
                mean    min    max  count
Department                             
Finance      70000.0  70000  70000      1
HR           52500.0  50000  55000      2
IT           63500.0  62000  65000      2

Multiple statistics by Department:
            Salary         Age Experience     
              mean    sum mean        min max
Department                                   
Finance      70000  70000   35          8   8
HR           52500 105000   28          2   6
IT           63500 127000   29          4   5
```

## Data Transformation

### Applying Functions

```python
# Apply function to a single column
df['Salary_Category'] = df['Salary'].apply(lambda x: 'High' if x > 60000 else 'Low')
print("Salary categories:")
print(df[['Name', 'Salary', 'Salary_Category']])

# Apply custom function
def categorize_experience(years):
    if years < 3:
        return 'Entry Level'
    elif years < 6:
        return 'Mid Level'
    else:
        return 'Senior Level'

df['Exp_Category'] = df['Experience'].apply(categorize_experience)
print("\nExperience categories:")
print(df[['Name', 'Experience', 'Exp_Category']])
```
**Output:**
```
Salary categories:
      Name  Salary Salary_Category
0    Alice   50000             Low
1      Bob   65000            High
2  Charlie   70000            High
3    Diana   62000            High
4      Eve   55000             Low

Experience categories:
      Name  Experience  Exp_Category
0    Alice           2   Entry Level
1      Bob           5     Mid Level
2  Charlie           8  Senior Level
3    Diana           4     Mid Level
4      Eve           6  Senior Level
```

```python
# Apply function to multiple columns
def calculate_value_score(row):
    return (row['Experience'] * 1000) + (row['Age'] * 100)

df['Value_Score'] = df.apply(calculate_value_score, axis=1)
print("Value scores:")
print(df[['Name', 'Age', 'Experience', 'Value_Score']])

# Apply function to entire DataFrame
numeric_cols = ['Age', 'Salary', 'Experience']
df_normalized = df[numeric_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
print("\nNormalized numeric columns:")
print(df_normalized.round(3))
```
**Output:**
```
Value scores:
      Name  Age  Experience  Value_Score
0    Alice   25           2         4500
1      Bob   30           5         8000
2  Charlie   35           8        11500
3    Diana   28           4         6800
4      Eve   32           6         9200

Normalized numeric columns:
    Age  Salary  Experience
0  0.000   0.000       0.000
1  0.500   0.750       0.500
2  1.000   1.000       1.000
3  0.300   0.600       0.333
4  0.700   0.250       0.667
```

### Mathematical Operations

```python
# Element-wise operations
df['Salary_Thousands'] = df['Salary'] / 1000
df['Age_Squared'] = df['Age'] ** 2
df['Log_Salary'] = np.log(df['Salary'])

print("Mathematical transformations:")
print(df[['Name', 'Salary', 'Salary_Thousands', 'Age_Squared', 'Log_Salary']].round(2))

# Operations between columns
df['Salary_Per_Year_Exp'] = df['Salary'] / df['Experience']
df['Age_Experience_Ratio'] = df['Age'] / df['Experience']

print("\nRatio calculations:")
print(df[['Name', 'Salary_Per_Year_Exp', 'Age_Experience_Ratio']].round(2))
```
**Output:**
```
Mathematical transformations:
      Name  Salary  Salary_Thousands  Age_Squared  Log_Salary
0    Alice   50000              50.0          625       10.82
1      Bob   65000              65.0          900       11.08
2  Charlie   70000              70.0         1225       11.16
3    Diana   62000              62.0          784       11.03
4      Eve   55000              55.0         1024       10.91

Ratio calculations:
      Name  Salary_Per_Year_Exp  Age_Experience_Ratio
0    Alice                25000                 12.50
1      Bob                13000                  6.00
2  Charlie                 8750                  4.38
3    Diana                15500                  7.00
4      Eve                 9167                  5.33
```

## Data Type Conversion

```python
# Check current data types
print("Current data types:")
print(df.dtypes)

# Convert data types
df_converted = df.copy()
df_converted['Age'] = df_converted['Age'].astype(float)
df_converted['Experience'] = df_converted['Experience'].astype('int32')
df_converted['Department'] = df_converted['Department'].astype('category')

print("\nAfter type conversion:")
print(df_converted.dtypes)
```
**Output:**
```
Current data types:
Name                     object
Age                       int64
Department               object
Salary                    int64
Experience                int64
Country                  object
Annual_Bonus            float64
Seniority                object
Age_Group                object
Salary_Category          object
Exp_Category             object
Value_Score               int64
Salary_Thousands        float64
Age_Squared               int64
Log_Salary              float64
Salary_Per_Year_Exp     float64
Age_Experience_Ratio    float64
dtype: object

After type conversion:
Name                     object
Age                     float64
Department             category
Salary                    int64
Experience                int32
Country                  object
Annual_Bonus            float64
Seniority                object
Age_Group                object
Salary_Category          object
Exp_Category             object
Value_Score               int64
Salary_Thousands        float64
Age_Squared               int64
Log_Salary              float64
Salary_Per_Year_Exp     float64
Age_Experience_Ratio    float64
dtype: object
```

```python
# Memory usage comparison
print("Memory usage comparison:")
print("Original:")
print(df.memory_usage(deep=True))
print("\nAfter conversion:")
print(df_converted.memory_usage(deep=True))

# Category data type benefits
print(f"\nDepartment column memory usage:")
print(f"Object: {df['Department'].memory_usage(deep=True)} bytes")
print(f"Category: {df_converted['Department'].memory_usage(deep=True)} bytes")
```
**Output:**
```
Memory usage comparison:
Original:
Index                      128
Name                       280
Age                         40
Department                 280
Salary                      40
Experience                  40
Country                    280
Annual_Bonus                40
Seniority                  280
Age_Group                  280
Salary_Category            280
Exp_Category               280
Value_Score                 40
Salary_Thousands            40
Age_Squared                 40
Log_Salary                  40
Salary_Per_Year_Exp         40
Age_Experience_Ratio        40
dtype: int64

After conversion:
Index                      128
Name                       280
Age                         40
Department                 293
Salary                      40
Experience                  20
Country                    280
Annual_Bonus                40
Seniority                  280
Age_Group                  280
Salary_Category            280
Exp_Category               280
Value_Score                 40
Salary_Thousands            40
Age_Squared                 40
Log_Salary                  40
Salary_Per_Year_Exp         40
Age_Experience_Ratio        40
dtype: int64

Department column memory usage:
Object: 280 bytes
Category: 293 bytes
```

## Summary of Key Operations

| Operation | Method | Example |
|-----------|--------|---------|
| Add column | `df['new_col'] = values` | `df['Total'] = df['A'] + df['B']` |
| Remove column | `df.drop()` | `df.drop(['col'], axis=1)` |
| Sort by values | `df.sort_values()` | `df.sort_values('Age')` |
| Sort by index | `df.sort_index()` | `df.sort_index()` |
| Basic stats | `df.describe()` | `df['Salary'].mean()` |
| Apply function | `df.apply()` | `df['Col'].apply(func)` |
| Group statistics | `df.groupby().agg()` | `df.groupby('Dept').mean()` |
| Type conversion | `df.astype()` | `df['Col'].astype('category')` |

### Best Practices

1. **Use vectorized operations** instead of loops
2. **Choose appropriate data types** for memory efficiency
3. **Use meaningful column names** for new columns
4. **Consider using .copy()** when modifying DataFrames
5. **Handle missing values** before mathematical operations

---

**Level 1 Complete! Next: Level 2 - Intermediate Data Manipulation**