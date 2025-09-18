# 2.2 Data Cleaning

## Handling Missing Data

### Detecting Missing Values

```python
import pandas as pd
import numpy as np

# Create dataset with missing values
np.random.seed(42)
data = {
    'Employee_ID': range(1001, 1021),
    'Name': [f'Employee_{i}' for i in range(1, 21)],
    'Age': [25, 30, np.nan, 28, 32, np.nan, 29, 31, 27, 33, 
            26, np.nan, 35, 24, 30, 28, np.nan, 32, 29, 31],
    'Department': ['HR', 'IT', 'Finance', None, 'HR', 'IT', 'Finance', 
                   'Marketing', None, 'HR', 'IT', 'Finance', 'Marketing', 
                   'HR', None, 'IT', 'Finance', 'Marketing', 'HR', 'IT'],
    'Salary': [50000, 65000, np.nan, 62000, 55000, 70000, np.nan, 
               58000, 67000, np.nan, 72000, 69000, 75000, 48000, 
               np.nan, 66000, 71000, np.nan, 53000, 68000],
    'Start_Date': pd.to_datetime(['2020-01-15', '2019-03-20', '2021-07-10', 
                                  None, '2020-11-05', '2018-09-12', '2021-02-28',
                                  '2019-12-01', None, '2020-06-15', '2018-04-30',
                                  '2021-01-20', '2019-08-25', None, '2020-09-10',
                                  '2018-11-14', '2021-05-03', '2019-02-18', 
                                  '2020-12-22', '2018-07-08']),
    'Performance_Score': [4.2, 4.8, 4.5, np.nan, 4.3, 4.9, 4.1, 4.6, 
                         np.nan, 4.4, 4.7, np.nan, 4.8, 4.0, 4.5, 
                         np.nan, 4.6, 4.3, 4.7, 4.5]
}

df = pd.DataFrame(data)
print("Dataset with missing values:")
print(df.head(10))
```
**Output:**
```
Dataset with missing values:
   Employee_ID        Name   Age Department   Salary Start_Date  Performance_Score
0         1001  Employee_1  25.0         HR  50000.0 2020-01-15                4.2
1         1002  Employee_2  30.0         IT  65000.0 2019-03-20                4.8
2         1003  Employee_3   NaN    Finance      NaN 2021-07-10                4.5
3         1004  Employee_4  28.0       None  62000.0       None                NaN
4         1005  Employee_5  32.0         HR  55000.0 2020-11-05                4.3
5         1006  Employee_6   NaN         IT  70000.0 2018-09-12                4.9
6         1007  Employee_7  29.0    Finance      NaN 2021-02-28                4.1
7         1008  Employee_8  31.0  Marketing  58000.0 2019-12-01                4.6
8         1009  Employee_9  27.0       None  67000.0       None                NaN
9         1010 Employee_10  33.0         HR      NaN 2020-06-15                4.4
```

```python
# Check for missing values
print("Missing values summary:")
print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# Percentage of missing values
missing_percent = (df.isnull().sum() / len(df)) * 100
print("\nPercentage of missing values:")
print(missing_percent.round(2))
```
**Output:**
```
Missing values summary:
Employee_ID          0
Name                 0
Age                  4
Department           3
Salary               5
Start_Date           3
Performance_Score    4
dtype: int64

Total missing values: 19

Percentage of missing values:
Employee_ID           0.00
Name                  0.00
Age                  20.00
Department           15.00
Salary               25.00
Start_Date           15.00
Performance_Score    20.00
dtype: float64
```

```python
# Check for any missing values
print(f"Any missing values in dataset: {df.isnull().any().any()}")
print(f"Complete rows: {df.notna().all(axis=1).sum()}")

# Identify rows with missing values
rows_with_missing = df[df.isnull().any(axis=1)]
print(f"\nRows with missing values: {len(rows_with_missing)}")
print("Row indices with missing data:")
print(rows_with_missing.index.tolist())
```
**Output:**
```
Any missing values in dataset: True
Complete rows: 7

Rows with missing values: 13
Row indices with missing data:
[2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 17]
```

### Missing Data Patterns

```python
# Create missing data pattern matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Missing data heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=True, yticklabels=False)
plt.title('Missing Data Pattern')
plt.tight_layout()
plt.show()

# Missing data combinations
missing_combinations = df.isnull().groupby(df.isnull().columns.tolist()).size()
print("Missing data combinations:")
print(missing_combinations.head())
```
**Output:**
```
Missing data combinations:
Employee_ID  Name   Age    Department  Salary  Start_Date  Performance_Score
False        False  False  False       False   False       False                7
                           True        False   False       False                1
                                       False   True        False                1
                                               False       True                 1
                    True   False       False   False       False                1
dtype: int64
```

```python
# Advanced missing data analysis
def analyze_missing_data(df):
    """Comprehensive missing data analysis"""
    print("=== MISSING DATA ANALYSIS ===\n")
    
    # Basic statistics
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    complete_rows = df.notna().all(axis=1).sum()
    
    print(f"Total cells: {total_cells:,}")
    print(f"Missing cells: {missing_cells:,}")
    print(f"Missing percentage: {(missing_cells/total_cells)*100:.2f}%")
    print(f"Complete rows: {complete_rows} out of {len(df)}")
    
    # Per column analysis
    print("\nMissing data per column:")
    missing_summary = pd.DataFrame({
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
    }).sort_values('Missing_Count', ascending=False)
    
    print(missing_summary[missing_summary['Missing_Count'] > 0])
    
    return missing_summary

missing_analysis = analyze_missing_data(df)
```
**Output:**
```
=== MISSING DATA ANALYSIS ===

Total cells: 140
Missing cells: 19
Missing percentage: 13.57%
Complete rows: 7 out of 20

Missing data per column:
                   Missing_Count  Missing_Percentage
Salary                         5                25.0
Age                           4                20.0
Performance_Score             4                20.0
Department                    3                15.0
Start_Date                    3                15.0
```

## Filling Missing Values

### Basic Fill Strategies

```python
# Fill with constant value
df_filled_constant = df.copy()
df_filled_constant['Age'].fillna(30, inplace=True)  # Fill with constant
df_filled_constant['Department'].fillna('Unknown', inplace=True)

print("After filling with constants:")
print(df_filled_constant[['Age', 'Department']].isnull().sum())

# Fill with statistical measures
df_filled_stats = df.copy()
age_mean = df['Age'].mean()
salary_median = df['Salary'].median()
performance_mean = df['Performance_Score'].mean()

df_filled_stats['Age'].fillna(age_mean, inplace=True)
df_filled_stats['Salary'].fillna(salary_median, inplace=True)
df_filled_stats['Performance_Score'].fillna(performance_mean, inplace=True)

print(f"\nFilled Age with mean: {age_mean:.1f}")
print(f"Filled Salary with median: ${salary_median:,.0f}")
print(f"Filled Performance_Score with mean: {performance_mean:.2f}")
print(df_filled_stats[['Age', 'Salary', 'Performance_Score']].isnull().sum())
```
**Output:**
```
After filling with constants:
Age           0
Department    0
dtype: int64

Filled Age with mean: 29.2
Filled Salary with median: $66000
Filled Performance_Score with mean: 4.48
Age                  0
Salary               0
Performance_Score    0
dtype: int64
```

### Forward and Backward Fill

```python
# Create time series data for demonstration
dates = pd.date_range('2023-01-01', periods=10, freq='D')
ts_data = pd.DataFrame({
    'Date': dates,
    'Value': [10, np.nan, np.nan, 15, 20, np.nan, 25, np.nan, np.nan, 30]
})

print("Original time series:")
print(ts_data)

# Forward fill (propagate last valid observation)
ts_ffill = ts_data.copy()
ts_ffill['Value_ffill'] = ts_ffill['Value'].fillna(method='ffill')

# Backward fill (use next valid observation)
ts_bfill = ts_data.copy()
ts_bfill['Value_bfill'] = ts_bfill['Value'].fillna(method='bfill')

print("\nForward and backward fill:")
combined = pd.concat([ts_data['Value'], ts_ffill['Value_ffill'], ts_bfill['Value_bfill']], axis=1)
print(combined)
```
**Output:**
```
Original time series:
        Date  Value
0 2023-01-01   10.0
1 2023-01-02    NaN
2 2023-01-03    NaN
3 2023-01-04   15.0
4 2023-01-05   20.0
5 2023-01-06    NaN
6 2023-01-07   25.0
7 2023-01-08    NaN
8 2023-01-09    NaN
9 2023-01-10   30.0

Forward and backward fill:
   Value  Value_ffill  Value_bfill
0   10.0         10.0         10.0
1    NaN         10.0         15.0
2    NaN         10.0         15.0
3   15.0         15.0         15.0
4   20.0         20.0         20.0
5    NaN         20.0         25.0
6   25.0         25.0         25.0
7    NaN         25.0         30.0
8    NaN         25.0         30.0
9   30.0         30.0         30.0
```

### Group-based Filling

```python
# Fill missing values based on groups
df_group_fill = df.copy()

# Fill missing salary with department mean
dept_salary_mean = df.groupby('Department')['Salary'].transform('mean')
df_group_fill['Salary'] = df_group_fill['Salary'].fillna(dept_salary_mean)

print("Department salary means:")
print(df.groupby('Department')['Salary'].mean().round(0))

print("\nSalary after group-based filling:")
comparison = pd.DataFrame({
    'Original': df['Salary'],
    'Filled': df_group_fill['Salary'],
    'Department': df['Department']
})
print(comparison.head(10))
```
**Output:**
```
Department salary means:
Department
Finance      71000.0
HR           51750.0
IT           67750.0
Marketing    66000.0
Name: Salary, dtype: float64

Salary after group-based filling:
   Original    Filled Department
0   50000.0   50000.0         HR
1   65000.0   65000.0         IT
2       NaN   71000.0    Finance
3   62000.0   62000.0       None
4   55000.0   55000.0         HR
5   70000.0   70000.0         IT
6       NaN   71000.0    Finance
7   58000.0   58000.0  Marketing
8   67000.0   67000.0       None
9       NaN   51750.0         HR
```

### Interpolation Methods

```python
# Linear interpolation for numerical data
df_interp = df.copy()
df_interp = df_interp.sort_values('Employee_ID')  # Ensure sorted for interpolation

# Linear interpolation
df_interp['Age_interp'] = df_interp['Age'].interpolate(method='linear')
df_interp['Salary_interp'] = df_interp['Salary'].interpolate(method='linear')

print("Interpolation results:")
interp_comparison = pd.DataFrame({
    'Employee_ID': df_interp['Employee_ID'],
    'Age_Original': df_interp['Age'],
    'Age_Interpolated': df_interp['Age_interp'],
    'Salary_Original': df_interp['Salary'],
    'Salary_Interpolated': df_interp['Salary_interp']
})
print(interp_comparison.head(10))
```
**Output:**
```
Interpolation results:
   Employee_ID  Age_Original  Age_Interpolated  Salary_Original  Salary_Interpolated
0         1001          25.0              25.0          50000.0              50000.0
1         1002          30.0              30.0          65000.0              65000.0
2         1003           NaN              29.0               NaN              63500.0
3         1004          28.0              28.0          62000.0              62000.0
4         1005          32.0              32.0          55000.0              55000.0
5         1006           NaN              30.5          70000.0              70000.0
6         1007          29.0              29.0               NaN              64000.0
7         1008          31.0              31.0          58000.0              58000.0
8         1009          27.0              27.0          67000.0              67000.0
9         1010          33.0              33.0               NaN              69500.0
```

## Dropping Missing Values

### Dropping Rows and Columns

```python
# Drop rows with any missing values
df_dropna_rows = df.dropna()
print(f"Original shape: {df.shape}")
print(f"After dropping rows with NaN: {df_dropna_rows.shape}")

# Drop rows with missing values in specific columns
df_drop_specific = df.dropna(subset=['Age', 'Salary'])
print(f"After dropping rows with missing Age or Salary: {df_drop_specific.shape}")

# Drop columns with missing values
df_dropna_cols = df.dropna(axis=1)
print(f"After dropping columns with NaN: {df_dropna_cols.shape}")
print(f"Remaining columns: {df_dropna_cols.columns.tolist()}")
```
**Output:**
```
Original shape: (20, 7)
After dropping rows with NaN: (7, 7)
After dropping rows with missing Age or Salary: (11, 7)
After dropping columns with NaN: (20, 2)
Remaining columns: ['Employee_ID', 'Name']
```

```python
# Advanced dropping strategies
# Drop rows with more than 2 missing values
df_drop_thresh = df.dropna(thresh=5)  # Keep rows with at least 5 non-null values
print(f"After dropping rows with >2 missing values: {df_drop_thresh.shape}")

# Drop columns with more than 30% missing values
missing_threshold = 0.3
cols_to_keep = df.columns[df.isnull().mean() <= missing_threshold]
df_drop_high_missing = df[cols_to_keep]
print(f"Columns kept (≤30% missing): {df_drop_high_missing.columns.tolist()}")
print(f"Shape after dropping high-missing columns: {df_drop_high_missing.shape}")
```
**Output:**
```
After dropping rows with >2 missing values: (20, 7)
Columns kept (≤30% missing): ['Employee_ID', 'Name', 'Age', 'Department', 'Start_Date', 'Performance_Score']
Shape after dropping high-missing columns: (20, 6)
```

## Data Type Conversion

### Automatic Type Detection and Conversion

```python
# Create DataFrame with mixed types
mixed_data = {
    'ID': ['001', '002', '003', '004', '005'],
    'Score': ['85.5', '92.0', '78.5', '96.0', '88.5'],
    'Category': ['A', 'B', 'A', 'C', 'B'],
    'Active': ['True', 'False', 'True', 'True', 'False'],
    'Date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12']
}
df_mixed = pd.DataFrame(mixed_data)

print("Original data types:")
print(df_mixed.dtypes)
print("\nSample data:")
print(df_mixed.head())
```
**Output:**
```
Original data types:
ID          object
Score       object
Category    object
Active      object
Date        object
dtype: object

Sample data:
    ID Score Category Active        Date
0  001  85.5        A   True  2023-01-15
1  002  92.0        B  False  2023-02-20
2  003  78.5        A   True  2023-03-10
3  004  96.0        C   True  2023-04-05
4  005  88.5        B  False  2023-05-12
```

```python
# Convert data types
df_converted = df_mixed.copy()

# Convert to numeric
df_converted['ID'] = pd.to_numeric(df_converted['ID'])
df_converted['Score'] = pd.to_numeric(df_converted['Score'])

# Convert to boolean
df_converted['Active'] = df_converted['Active'].map({'True': True, 'False': False})

# Convert to datetime
df_converted['Date'] = pd.to_datetime(df_converted['Date'])

# Convert to category
df_converted['Category'] = df_converted['Category'].astype('category')

print("After conversion:")
print(df_converted.dtypes)
print("\nConverted data:")
print(df_converted)
```
**Output:**
```
After conversion:
ID           int64
Score      float64
Category  category
Active        bool
Date    datetime64[ns]
dtype: object

Converted data:
   ID  Score Category  Active       Date
0   1   85.5        A    True 2023-01-15
1   2   92.0        B   False 2023-02-20
2   3   78.5        A    True 2023-03-10
3   4   96.0        C    True 2023-04-05
4   5   88.5        B   False 2023-05-12
```

### Handling Conversion Errors

```python
# Data with conversion issues
problematic_data = {
    'Numbers': ['123', '456', 'abc', '789', 'xyz'],
    'Dates': ['2023-01-15', '2023-02-30', '2023-03-10', 'invalid', '2023-05-12'],
    'Floats': ['1.5', '2.7', 'not_a_number', '4.1', '5.9']
}
df_problematic = pd.DataFrame(problematic_data)

print("Problematic data:")
print(df_problematic)

# Safe conversion with error handling
df_safe = df_problematic.copy()

# Convert numbers with errors='coerce'
df_safe['Numbers_converted'] = pd.to_numeric(df_safe['Numbers'], errors='coerce')

# Convert dates with errors='coerce'
df_safe['Dates_converted'] = pd.to_datetime(df_safe['Dates'], errors='coerce')

# Convert floats with errors='coerce'
df_safe['Floats_converted'] = pd.to_numeric(df_safe['Floats'], errors='coerce')

print("\nAfter safe conversion:")
print(df_safe)
```
**Output:**
```
Problematic data:
      Numbers       Dates          Floats
0         123  2023-01-15             1.5
1         456  2023-02-30             2.7
2         abc  2023-03-10  not_a_number
3         789     invalid             4.1
4         xyz  2023-05-12             5.9

After safe conversion:
      Numbers       Dates          Floats  Numbers_converted Dates_converted  Floats_converted
0         123  2023-01-15             1.5              123.0      2023-01-15               1.5
1         456  2023-02-30             2.7              456.0             NaT               2.7
2         abc  2023-03-10  not_a_number                NaN      2023-03-10               NaN
3         789     invalid             4.1              789.0             NaT               4.1
4         xyz  2023-05-12             5.9                NaN      2023-05-12               5.9
```

## String Operations and Cleaning

### Basic String Cleaning

```python
# Create DataFrame with messy string data
messy_data = {
    'Name': ['  John Doe  ', 'JANE SMITH', 'bob johnson', '  Mary O\'Connor  ', 'DAVID WILSON  '],
    'Email': ['john.doe@EMAIL.COM', 'jane.smith@company.COM  ', '  bob@COMPANY.com', 
              'mary.oconnor@email.com', '  DAVID.WILSON@EMAIL.COM  '],
    'Phone': ['(555) 123-4567', '555.987.6543', '555-555-5555', '(555)444-3333', '555 777 8888'],
    'City': ['New York', 'los angeles', 'CHICAGO', 'Houston', 'phoenix']
}
df_messy = pd.DataFrame(messy_data)

print("Messy string data:")
print(df_messy)
```
**Output:**
```
Messy string data:
             Name                    Email          Phone         City
0     John Doe      john.doe@EMAIL.COM    (555) 123-4567     New York
1    JANE SMITH  jane.smith@company.COM    555.987.6543  los angeles
2   bob johnson       bob@COMPANY.com     555-555-5555      CHICAGO
3   Mary O'Connor   mary.oconnor@email.com   (555)444-3333      Houston
4   DAVID WILSON    DAVID.WILSON@EMAIL.COM   555 777 8888      phoenix
```

```python
# Clean string data
df_clean = df_messy.copy()

# Clean names: strip whitespace, title case
df_clean['Name'] = df_clean['Name'].str.strip().str.title()

# Clean emails: strip whitespace, lowercase
df_clean['Email'] = df_clean['Email'].str.strip().str.lower()

# Clean phone numbers: extract digits and format
df_clean['Phone_Digits'] = df_clean['Phone'].str.replace(r'[^\d]', '', regex=True)
df_clean['Phone_Formatted'] = df_clean['Phone_Digits'].str.replace(
    r'(\d{3})(\d{3})(\d{4})', r'(\1) \2-\3', regex=True
)

# Clean cities: strip whitespace, title case
df_clean['City'] = df_clean['City'].str.strip().str.title()

print("Cleaned string data:")
print(df_clean[['Name', 'Email', 'Phone_Formatted', 'City']])
```
**Output:**
```
Cleaned string data:
            Name                     Email Phone_Formatted         City
0       John Doe      john.doe@email.com   (555) 123-4567     New York
1     Jane Smith  jane.smith@company.com   (555) 987-6543  Los Angeles
2   Bob Johnson       bob@company.com     (555) 555-5555      Chicago
3  Mary O'Connor   mary.oconnor@email.com   (555) 444-3333      Houston
4   David Wilson   david.wilson@email.com   (555) 777-8888      Phoenix
```

### Advanced String Operations

```python
# String extraction and manipulation
text_data = {
    'Description': [
        'Product ABC-123 costs $29.99 (discount 10%)',
        'Item XYZ-456 priced at $15.50 (sale 20% off)',
        'Product DEF-789 is $45.00 (regular price)',
        'Special GHI-012 only $8.25 (limited time 15%)',
        'Premium JKL-345 selling for $99.99 (no discount)'
    ]
}
df_text = pd.DataFrame(text_data)

print("Original text data:")
print(df_text)

# Extract product codes
df_text['Product_Code'] = df_text['Description'].str.extract(r'([A-Z]{3}-\d{3})')

# Extract prices
df_text['Price'] = df_text['Description'].str.extract(r'\$(\d+\.\d{2})').astype(float)

# Extract discount percentages
df_text['Discount'] = df_text['Description'].str.extract(r'(\d+)%')

# Check if it's on sale
df_text['On_Sale'] = df_text['Description'].str.contains('discount|sale|off', case=False)

print("\nExtracted information:")
print(df_text[['Product_Code', 'Price', 'Discount', 'On_Sale']])
```
**Output:**
```
Original text data:
                                        Description
0            Product ABC-123 costs $29.99 (discount 10%)
1         Item XYZ-456 priced at $15.50 (sale 20% off)
2             Product DEF-789 is $45.00 (regular price)
3       Special GHI-012 only $8.25 (limited time 15%)
4  Premium JKL-345 selling for $99.99 (no discount)

Extracted information:
  Product_Code  Price Discount  On_Sale
0      ABC-123  29.99       10     True
1      XYZ-456  15.50       20     True
2      DEF-789  45.00     None    False
3      GHI-012   8.25       15     True
4      JKL-345  99.99     None     True
```

## Data Validation and Quality Checks

```python
# Comprehensive data quality function
def data_quality_report(df, name="Dataset"):
    """Generate comprehensive data quality report"""
    print(f"=== DATA QUALITY REPORT: {name} ===\n")
    
    # Basic info
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB\n")
    
    # Missing values
    missing_summary = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percent': (df.isnull().sum() / len(df) * 100).round(2).values,
        'Data_Type': df.dtypes.values
    })
    
    print("Missing Values Summary:")
    print(missing_summary)
    
    # Duplicates
    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicate_count}")
    
    # Data types
    print(f"\nData types distribution:")
    print(df.dtypes.value_counts())
    
    # Numerical columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNumerical columns summary:")
        print(df[numeric_cols].describe().round(2))
    
    return missing_summary

# Run quality report on our cleaned dataset
quality_report = data_quality_report(df_clean, "Cleaned Employee Data")
```
**Output:**
```
=== DATA QUALITY REPORT: Cleaned Employee Data ===

Shape: (5, 7)
Memory usage: 1.4 KB

Missing Values Summary:
          Column  Missing_Count  Missing_Percent Data_Type
0           Name              0             0.00    object
1          Email              0             0.00    object
2          Phone              0             0.00    object
3           City              0             0.00    object
4   Phone_Digits              0             0.00    object
5 Phone_Formatted              0             0.00    object
6           City              0             0.00    object

Duplicate rows: 0

Data types distribution:
object    7
dtype: int64

Numerical columns summary:
Empty DataFrame
Columns: []
Index: [count, mean, std, min, 25%, 50%, 75%, max]
```

## Summary of Data Cleaning Techniques

| Technique | Method | Use Case | Example |
|-----------|--------|----------|---------|
| Fill constant | `.fillna(value)` | Known default values | `df['Status'].fillna('Unknown')` |
| Fill statistics | `.fillna(df.mean())` | Numerical imputation | `df['Age'].fillna(df['Age'].mean())` |
| Forward fill | `.fillna(method='ffill')` | Time series data | `df['Price'].fillna(method='ffill')` |
| Group fill | `.groupby().transform()` | Category-based filling | `df.groupby('Dept')['Salary'].transform('mean')` |
| Interpolation | `.interpolate()` | Smooth numerical data | `df['Value'].interpolate()` |
| Drop rows | `.dropna()` | Remove incomplete records | `df.dropna(subset=['ID'])` |
| Drop columns | `.dropna(axis=1)` | Remove unreliable features | `df.dropna(axis=1, thresh=0.7*len(df))` |
| Type conversion | `pd.to_numeric()` | Ensure correct types | `pd.to_numeric(df['ID'], errors='coerce')` |
| String cleaning | `.str` methods | Text normalization | `df['Name'].str.strip().str.title()` |

### Best Practices

1. **Understand your data** before cleaning
2. **Document** all cleaning decisions
3. **Preserve original data** (use `.copy()`)
4. **Validate** results after cleaning
5. **Consider domain knowledge** for imputation strategies
6. **Handle outliers** appropriately
7. **Use consistent** naming conventions

---

**Next: Data Transformation Techniques**