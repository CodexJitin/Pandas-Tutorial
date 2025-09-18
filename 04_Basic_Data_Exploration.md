# 1.4 Basic Data Exploration

## Viewing Data

### First Look at Data

```python
import pandas as pd
import numpy as np

# Create sample dataset
np.random.seed(42)
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry'],
    'Age': [25, 30, 35, 28, 32, 27, 29, 31],
    'Department': ['HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'IT', 'HR'],
    'Salary': [50000, 65000, 70000, 62000, 55000, 68000, 63000, 52000],
    'Experience': [2, 5, 8, 4, 6, 3, 4, 7],
    'Rating': [4.2, 4.8, 4.5, 4.7, 4.3, 4.6, 4.9, 4.1]
}
df = pd.DataFrame(data)

# View first few rows
print("First 5 rows:")
print(df.head())
```
**Output:**
```
First 5 rows:
      Name  Age Department  Salary  Experience  Rating
0    Alice   25         HR   50000           2     4.2
1      Bob   30         IT   65000           5     4.8
2  Charlie   35    Finance   70000           8     4.5
3    Diana   28         IT   62000           4     4.7
4      Eve   32         HR   55000           6     4.3
```

```python
# View last few rows
print("Last 3 rows:")
print(df.tail(3))
```
**Output:**
```
Last 3 rows:
    Name  Age Department  Salary  Experience  Rating
5  Frank   27    Finance   68000           3     4.6
6  Grace   29         IT   63000           4     4.9
7  Henry   31         HR   52000           7     4.1
```

```python
# View specific number of rows
print("First 3 rows:")
print(df.head(3))

print("\nLast 2 rows:")
print(df.tail(2))
```
**Output:**
```
First 3 rows:
      Name  Age Department  Salary  Experience  Rating
0    Alice   25         HR   50000           2     4.2
1      Bob   30         IT   65000           5     4.8
2  Charlie   35    Finance   70000           8     4.5

Last 2 rows:
    Name  Age Department  Salary  Experience  Rating
6  Grace   29         IT   63000           4     4.9
7  Henry   31         HR   52000           7     4.1
```

### Random Sampling

```python
# Random sample of rows
print("Random sample of 3 rows:")
print(df.sample(3))
```
**Output:**
```
Random sample of 3 rows:
    Name  Age Department  Salary  Experience  Rating
2  Charlie   35    Finance   70000           8     4.5
5    Frank   27    Finance   68000           3     4.6
0    Alice   25         HR   50000           2     4.2
```

```python
# Random sample with seed for reproducibility
print("Random sample with seed:")
print(df.sample(2, random_state=42))

# Sample by fraction
print("\nSample 25% of data:")
print(df.sample(frac=0.25, random_state=42))
```
**Output:**
```
Random sample with seed:
    Name  Age Department  Salary  Experience  Rating
6  Grace   29         IT   63000           4     4.9
3  Diana   28         IT   62000           4     4.7

Sample 25% of data:
    Name  Age Department  Salary  Experience  Rating
6  Grace   29         IT   63000           4     4.9
3  Diana   28         IT   62000           4     4.7
```

## Basic Information About Data

### Data Structure Information

```python
# Basic info about DataFrame
print("DataFrame Info:")
print(df.info())
```
**Output:**
```
DataFrame Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8 entries, 0 to 7
Data columns (total 6 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   Name        8 non-null      object 
 1   Age         8 non-null      int64  
 2   Department  8 non-null      object 
 3   Salary      8 non-null      int64  
 4   Experience  8 non-null      int64  
 5   Rating      8 non-null      float64
dtypes: float64(1), int64(3), object(2)
memory usage: 512.0+ bytes
```

```python
# Data types
print("Data types:")
print(df.dtypes)
print(f"\nTotal memory usage: {df.memory_usage(deep=True).sum()} bytes")
```
**Output:**
```
Data types:
Name          object
Age            int64
Department    object
Salary         int64
Experience     int64
Rating       float64
dtype: object

Total memory usage: 712 bytes
```

```python
# Shape and dimensions
print(f"Shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"Size (total elements): {df.size}")
print(f"Dimensions: {df.ndim}")
```
**Output:**
```
Shape: (8, 6)
Number of rows: 8
Number of columns: 6
Size (total elements): 48
Dimensions: 2
```

### Column and Index Information

```python
# Column names
print("Columns:")
print(df.columns.tolist())

# Index information
print(f"\nIndex: {df.index}")
print(f"Index type: {type(df.index)}")

# Column and index names
print(f"\nColumn names: {list(df.columns)}")
print(f"Index name: {df.index.name}")
```
**Output:**
```
Columns:
['Name', 'Age', 'Department', 'Salary', 'Experience', 'Rating']

Index: RangeIndex(start=0, stop=8, step=1)
Index type: <class 'pandas.core.indexes.range.RangeIndex'>

Column names: ['Name', 'Age', 'Department', 'Salary', 'Experience', 'Rating']
Index name: None
```

## Statistical Summary

### Descriptive Statistics

```python
# Descriptive statistics for numerical columns
print("Descriptive statistics:")
print(df.describe())
```
**Output:**
```
Descriptive statistics:
             Age        Salary   Experience      Rating
count   8.000000      8.000000     8.000000    8.000000
mean   29.625000  59375.000000     4.875000    4.512500
std     3.114482   7440.831456     2.295184    0.309839
min    25.000000  50000.000000     2.000000    4.100000
25%    27.500000  54000.000000     3.250000    4.275000
50%    29.500000  62500.000000     4.500000    4.550000
75%    31.250000  65750.000000     6.250000    4.725000
max    35.000000  70000.000000     8.000000    4.900000
```

```python
# Include all columns (including non-numeric)
print("Descriptive statistics for all columns:")
print(df.describe(include='all'))
```
**Output:**
```
Descriptive statistics for all columns:
         Name        Age Department        Salary   Experience      Rating
count       8   8.000000          8      8.000000     8.000000    8.000000
unique      8        NaN          3           NaN          NaN         NaN
top     Alice        NaN         IT           NaN          NaN         NaN
freq        1        NaN          3           NaN          NaN         NaN
mean      NaN  29.625000        NaN  59375.000000     4.875000    4.512500
std       NaN   3.114482        NaN   7440.831456     2.295184    0.309839
min       NaN  25.000000        NaN  50000.000000     2.000000    4.100000
25%       NaN  27.500000        NaN  54000.000000     3.250000    4.275000
50%       NaN  29.500000        NaN  62500.000000     4.500000    4.550000
75%       NaN  31.250000        NaN  65750.000000     6.250000    4.725000
max       NaN  35.000000        NaN  70000.000000     8.000000    4.900000
```

```python
# Specific statistics
print("Individual statistics:")
print(f"Mean age: {df['Age'].mean():.1f}")
print(f"Median salary: ${df['Salary'].median():,}")
print(f"Standard deviation of rating: {df['Rating'].std():.3f}")
print(f"Min experience: {df['Experience'].min()} years")
print(f"Max experience: {df['Experience'].max()} years")
```
**Output:**
```
Individual statistics:
Mean age: 29.6
Median salary: $62,500
Standard deviation of rating: 0.310
Min experience: 2 years
Max experience: 8 years
```

### Value Counts

```python
# Count unique values in categorical columns
print("Department counts:")
print(df['Department'].value_counts())
```
**Output:**
```
Department counts:
Department
IT         3
HR         3
Finance    2
Name: count, dtype: int64
```

```python
# Value counts with percentages
print("Department distribution (percentages):")
print(df['Department'].value_counts(normalize=True) * 100)

# Value counts including missing values
print("\nDepartment counts (including NaN):")
print(df['Department'].value_counts(dropna=False))
```
**Output:**
```
Department distribution (percentages):
Department
IT         37.5
HR         37.5
Finance    25.0
Name: proportion, dtype: float64

Department counts (including NaN):
Department
IT         3
HR         3
Finance    2
Name: count, dtype: int64
```

```python
# Value counts for numerical data (binning)
print("Age groups:")
age_bins = pd.cut(df['Age'], bins=3, labels=['Young', 'Middle', 'Senior'])
print(age_bins.value_counts())
```
**Output:**
```
Age groups:
Age
Young     3
Middle    3
Senior    2
Name: count, dtype: int64
```

## Unique Values and Duplicates

### Unique Values

```python
# Unique values in each column
print("Unique values per column:")
for col in df.columns:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count} unique values")
```
**Output:**
```
Unique values per column:
Name: 8 unique values
Age: 8 unique values
Department: 3 unique values
Salary: 8 unique values
Experience: 7 unique values
Rating: 8 unique values
```

```python
# Get actual unique values
print("Unique departments:")
print(df['Department'].unique())

print("\nUnique ages (sorted):")
print(sorted(df['Age'].unique()))
```
**Output:**
```
Unique departments:
['HR' 'IT' 'Finance']

Unique ages (sorted):
[25, 27, 28, 29, 30, 31, 32, 35]
```

### Checking for Duplicates

```python
# Check for duplicate rows
print("Duplicate rows:")
print(f"Number of duplicate rows: {df.duplicated().sum()}")
print(f"Any duplicates? {df.duplicated().any()}")

# Add a duplicate row for demonstration
df_with_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
print(f"\nAfter adding duplicate: {df_with_dup.duplicated().sum()} duplicates")
print("Duplicate row indices:")
print(df_with_dup[df_with_dup.duplicated()].index.tolist())
```
**Output:**
```
Duplicate rows:
Number of duplicate rows: 0
Any duplicates? False

After adding duplicate: 1 duplicates
Duplicate row indices:
[8]
```

```python
# Check duplicates based on specific columns
print("Duplicates based on Department:")
dept_duplicates = df.duplicated(subset=['Department'], keep='first')
print(f"Count: {dept_duplicates.sum()}")
print("Rows with duplicate departments (keeping first):")
print(df[dept_duplicates][['Name', 'Department']])
```
**Output:**
```
Duplicates based on Department:
Count: 5
Rows with duplicate departments (keeping first):
    Name Department
1    Bob         IT
3  Diana         IT
4    Eve         HR
5  Frank    Finance
6  Grace         IT
7  Henry         HR
```

## Missing Data Overview

```python
# Add some missing values for demonstration
df_missing = df.copy()
df_missing.loc[1, 'Salary'] = np.nan
df_missing.loc[3, 'Rating'] = np.nan
df_missing.loc[5, 'Age'] = np.nan

# Check for missing values
print("Missing values per column:")
print(df_missing.isnull().sum())
```
**Output:**
```
Missing values per column:
Name          0
Age           1
Department    0
Salary        1
Experience    0
Rating        1
dtype: int64
```

```python
# Percentage of missing values
print("Percentage of missing values:")
missing_percent = (df_missing.isnull().sum() / len(df_missing)) * 100
print(missing_percent.round(2))

# Total missing values
print(f"\nTotal missing values: {df_missing.isnull().sum().sum()}")
print(f"Any missing values? {df_missing.isnull().any().any()}")
```
**Output:**
```
Percentage of missing values:
Name           0.00
Age           12.50
Department     0.00
Salary        12.50
Experience     0.00
Rating        12.50
dtype: float64

Total missing values: 3
Any missing values? True
```

```python
# Rows with any missing values
print("Rows with missing values:")
print(df_missing[df_missing.isnull().any(axis=1)])
```
**Output:**
```
Rows with missing values:
    Name   Age Department   Salary  Experience  Rating
1    Bob  30.0         IT      NaN           5     4.8
3  Diana  28.0         IT  62000.0           4     NaN
5  Frank   NaN    Finance  68000.0           3     4.6
```

## Data Quality Assessment

```python
# Comprehensive data quality report
def data_quality_report(df):
    print("=== DATA QUALITY REPORT ===\n")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total elements: {df.size}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum()} bytes\n")
    
    print("Missing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    for col in df.columns:
        print(f"  {col}: {missing[col]} ({missing_pct[col]:.1f}%)")
    
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    
    print("\nData types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    print("\nUnique values:")
    for col in df.columns:
        print(f"  {col}: {df[col].nunique()}")

# Run the report
data_quality_report(df)
```
**Output:**
```
=== DATA QUALITY REPORT ===

Dataset shape: (8, 6)
Total elements: 48
Memory usage: 712 bytes

Missing values:
  Name: 0 (0.0%)
  Age: 0 (0.0%)
  Department: 0 (0.0%)
  Salary: 0 (0.0%)
  Experience: 0 (0.0%)
  Rating: 0 (0.0%)

Duplicate rows: 0

Data types:
  Name: object
  Age: int64
  Department: object
  Salary: int64
  Experience: int64
  Rating: float64

Unique values:
  Name: 8
  Age: 8
  Department: 3
  Salary: 8
  Experience: 7
  Rating: 8
```

## Quick Exploration Functions

```python
# Custom exploration function
def quick_explore(df, sample_size=5):
    print("=== QUICK DATA EXPLORATION ===\n")
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    
    print("Sample data:")
    print(df.head(sample_size))
    print("\n" + "="*50 + "\n")
    
    print("Statistical summary:")
    print(df.describe())
    print("\n" + "="*50 + "\n")
    
    print("Missing values:")
    print(df.isnull().sum())
    print("\n" + "="*50 + "\n")
    
    print("Data types:")
    print(df.dtypes)

# Use the function
quick_explore(df, sample_size=3)
```
**Output:**
```
=== QUICK DATA EXPLORATION ===

Shape: (8, 6)
Columns: ['Name', 'Age', 'Department', 'Salary', 'Experience', 'Rating']

Sample data:
      Name  Age Department  Salary  Experience  Rating
0    Alice   25         HR   50000           2     4.2
1      Bob   30         IT   65000           5     4.8
2  Charlie   35    Finance   70000           8     4.5

==================================================

Statistical summary:
             Age        Salary   Experience      Rating
count   8.000000      8.000000     8.000000    8.000000
mean   29.625000  59375.000000     4.875000    4.512500
std     3.114482   7440.831456     2.295184    0.309839
min    25.000000  50000.000000     2.000000    4.100000
25%    27.500000  54000.000000     3.250000    4.275000
50%    29.500000  62500.000000     4.500000    4.550000
75%    31.250000  65750.000000     6.250000    4.725000
max    35.000000  70000.000000     8.000000    4.900000

==================================================

Missing values:
Name          0
Age           0
Department    0
Salary        0
Experience    0
Rating        0
dtype: int64

==================================================

Data types:
Name          object
Age            int64
Department    object
Salary         int64
Experience     int64
Rating       float64
dtype: object
```

---

**Next: Data Selection and Indexing**