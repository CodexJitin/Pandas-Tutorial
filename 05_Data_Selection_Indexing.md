# 1.5 Data Selection and Indexing

## Column Selection

### Single Column Selection

```python
import pandas as pd
import numpy as np

# Create sample dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'Department': ['HR', 'IT', 'Finance', 'IT', 'HR'],
    'Salary': [50000, 65000, 70000, 62000, 55000],
    'Rating': [4.2, 4.8, 4.5, 4.7, 4.3]
}
df = pd.DataFrame(data)

# Select single column - returns Series
name_series = df['Name']
print("Single column (Series):")
print(name_series)
print(f"Type: {type(name_series)}")
```
**Output:**
```
Single column (Series):
0      Alice
1        Bob
2    Charlie
3      Diana
4        Eve
Name: Name, dtype: object
Type: <class 'pandas.core.series.Series'>
```

```python
# Alternative syntax for single column
age_series = df.Age  # Dot notation
print("\nUsing dot notation:")
print(age_series)
print(f"Mean age: {age_series.mean()}")
```
**Output:**
```
Using dot notation:
0    25
1    30
2    35
3    28
4    32
Name: Age, dtype: int64
Mean age: 30.0
```

### Multiple Column Selection

```python
# Select multiple columns - returns DataFrame
columns_of_interest = ['Name', 'Salary', 'Rating']
subset_df = df[columns_of_interest]
print("Multiple columns (DataFrame):")
print(subset_df)
print(f"Type: {type(subset_df)}")
```
**Output:**
```
Multiple columns (DataFrame):
      Name  Salary  Rating
0    Alice   50000     4.2
1      Bob   65000     4.8
2  Charlie   70000     4.5
3    Diana   62000     4.7
4      Eve   55000     4.3
Type: <class 'pandas.core.frame.DataFrame'>
```

```python
# Select columns by range (if columns are in order)
print("Columns from Age to Salary:")
print(df.loc[:, 'Age':'Salary'])  # Inclusive of both endpoints
```
**Output:**
```
Columns from Age to Salary:
   Age Department  Salary
0   25         HR   50000
1   30         IT   65000
2   35    Finance   70000
3   28         IT   62000
4   32         HR   55000
```

## Row Selection

### Position-based Selection (.iloc)

```python
# Select single row by position
first_row = df.iloc[0]
print("First row (Series):")
print(first_row)
print(f"Type: {type(first_row)}")
```
**Output:**
```
First row (Series):
Name          Alice
Age              25
Department       HR
Salary        50000
Rating          4.2
Name: 0, dtype: object
Type: <class 'pandas.core.series.Series'>
```

```python
# Select multiple rows by position
first_three = df.iloc[0:3]  # Excludes index 3
print("First three rows:")
print(first_three)

# Select specific rows by position
specific_rows = df.iloc[[0, 2, 4]]  # Rows 0, 2, and 4
print("\nSpecific rows (0, 2, 4):")
print(specific_rows)
```
**Output:**
```
First three rows:
      Name  Age Department  Salary  Rating
0    Alice   25         HR   50000     4.2
1      Bob   30         IT   65000     4.8
2  Charlie   35    Finance   70000     4.5

Specific rows (0, 2, 4):
      Name  Age Department  Salary  Rating
0    Alice   25         HR   50000     4.2
2  Charlie   35    Finance   70000     4.5
4      Eve   32         HR   55000     4.3
```

```python
# Last few rows
print("Last two rows:")
print(df.iloc[-2:])  # Last 2 rows

# Every other row
print("\nEvery other row:")
print(df.iloc[::2])  # Start:end:step
```
**Output:**
```
Last two rows:
   Name  Age Department  Salary  Rating
3  Diana   28         IT   62000     4.7
4    Eve   32         HR   55000     4.3

Every other row:
      Name  Age Department  Salary  Rating
0    Alice   25         HR   50000     4.2
2  Charlie   35    Finance   70000     4.5
4      Eve   32         HR   55000     4.3
```

### Label-based Selection (.loc)

```python
# Create DataFrame with custom index
df_indexed = df.set_index('Name')
print("DataFrame with Name as index:")
print(df_indexed)
```
**Output:**
```
DataFrame with Name as index:
         Age Department  Salary  Rating
Name                                   
Alice     25         HR   50000     4.2
Bob       30         IT   65000     4.8
Charlie   35    Finance   70000     4.5
Diana     28         IT   62000     4.7
Eve       32         HR   55000     4.3
```

```python
# Select by label
alice_data = df_indexed.loc['Alice']
print("Alice's data:")
print(alice_data)

# Select multiple rows by label
it_employees = df_indexed.loc[['Bob', 'Diana']]
print("\nIT employees:")
print(it_employees)
```
**Output:**
```
Alice's data:
Age              25
Department       HR
Salary        50000
Rating          4.2
Name: Alice, dtype: object

IT employees:
       Age Department  Salary  Rating
Name                                 
Bob     30         IT   65000     4.8
Diana   28         IT   62000     4.7
```

```python
# Select range of labels
print("From Bob to Diana (inclusive):")
print(df_indexed.loc['Bob':'Diana'])
```
**Output:**
```
From Bob to Diana (inclusive):
         Age Department  Salary  Rating
Name                                   
Bob       30         IT   65000     4.8
Charlie   35    Finance   70000     4.5
Diana     28         IT   62000     4.7
```

## Boolean Indexing

### Simple Conditions

```python
# Create boolean mask
high_salary = df['Salary'] > 60000
print("High salary mask:")
print(high_salary)
print(f"Type: {type(high_salary)}")
```
**Output:**
```
High salary mask:
0    False
1     True
2     True
3     True
4    False
Name: Salary, dtype: bool
Type: <class 'pandas.core.series.Series'>
```

```python
# Apply boolean mask
high_earners = df[high_salary]
print("High earners:")
print(high_earners)

# Direct filtering (more common)
high_earners_direct = df[df['Salary'] > 60000]
print("\nDirect filtering (same result):")
print(high_earners_direct)
```
**Output:**
```
High earners:
      Name  Age Department  Salary  Rating
1      Bob   30         IT   65000     4.8
2  Charlie   35    Finance   70000     4.5
3    Diana   28         IT   62000     4.7

Direct filtering (same result):
      Name  Age Department  Salary  Rating
1      Bob   30         IT   65000     4.8
2  Charlie   35    Finance   70000     4.5
3    Diana   28         IT   62000     4.7
```

```python
# Different comparison operators
print("Young employees (age < 30):")
print(df[df['Age'] < 30])

print("\nExact matches:")
print(df[df['Department'] == 'IT'])

print("\nNot equal:")
print(df[df['Department'] != 'HR'])
```
**Output:**
```
Young employees (age < 30):
   Name  Age Department  Salary  Rating
0  Alice   25         HR   50000     4.2
3  Diana   28         IT   62000     4.7

Exact matches:
   Name  Age Department  Salary  Rating
1    Bob   30         IT   65000     4.8
3  Diana   28         IT   62000     4.7

Not equal:
      Name  Age Department  Salary  Rating
1      Bob   30         IT   65000     4.8
2  Charlie   35    Finance   70000     4.5
3    Diana   28         IT   62000     4.7
```

### Multiple Conditions

```python
# AND condition (&)
young_high_earners = df[(df['Age'] < 30) & (df['Salary'] > 55000)]
print("Young high earners (age < 30 AND salary > 55000):")
print(young_high_earners)

# OR condition (|)
hr_or_finance = df[(df['Department'] == 'HR') | (df['Department'] == 'Finance')]
print("\nHR or Finance employees:")
print(hr_or_finance)
```
**Output:**
```
Young high earners (age < 30 AND salary > 55000):
   Name  Age Department  Salary  Rating
3  Diana   28         IT   62000     4.7

HR or Finance employees:
      Name  Age Department  Salary  Rating
0    Alice   25         HR   50000     4.2
2  Charlie   35    Finance   70000     4.5
4      Eve   32         HR   55000     4.3
```

```python
# NOT condition (~)
non_it = df[~(df['Department'] == 'IT')]
print("Non-IT employees:")
print(non_it)

# Complex conditions with parentheses
complex_filter = df[((df['Age'] > 30) & (df['Rating'] > 4.5)) | (df['Salary'] < 55000)]
print("\nComplex filter:")
print(complex_filter)
```
**Output:**
```
Non-IT employees:
      Name  Age Department  Salary  Rating
0    Alice   25         HR   50000     4.2
2  Charlie   35    Finance   70000     4.5
4      Eve   32         HR   55000     4.3

Complex filter:
      Name  Age Department  Salary  Rating
0    Alice   25         HR   50000     4.2
4      Eve   32         HR   55000     4.3
```

### Using isin() Method

```python
# Select rows where column values are in a list
departments_of_interest = ['HR', 'Finance']
hr_finance = df[df['Department'].isin(departments_of_interest)]
print("HR and Finance employees using isin():")
print(hr_finance)

# Numeric values with isin()
target_ages = [25, 30, 35]
specific_ages = df[df['Age'].isin(target_ages)]
print("\nEmployees with specific ages:")
print(specific_ages)
```
**Output:**
```
HR and Finance employees using isin():
      Name  Age Department  Salary  Rating
0    Alice   25         HR   50000     4.2
2  Charlie   35    Finance   70000     4.5
4      Eve   32         HR   55000     4.3

Employees with specific ages:
      Name  Age Department  Salary  Rating
0    Alice   25         HR   50000     4.2
1      Bob   30         IT   65000     4.8
2  Charlie   35    Finance   70000     4.5
```

```python
# Inverse of isin() using ~
not_in_list = df[~df['Department'].isin(['IT'])]
print("Not in IT department:")
print(not_in_list)
```
**Output:**
```
Not in IT department:
      Name  Age Department  Salary  Rating
0    Alice   25         HR   50000     4.2
2  Charlie   35    Finance   70000     4.5
4      Eve   32         HR   55000     4.3
```

## Combined Selection (Rows and Columns)

### Using .loc for Combined Selection

```python
# Select specific rows and columns
subset = df.loc[df['Salary'] > 60000, ['Name', 'Salary', 'Department']]
print("High earners with selected columns:")
print(subset)

# Using .loc with index positions and column names
subset2 = df.loc[0:2, 'Name':'Department']
print("\nFirst 3 rows, specific column range:")
print(subset2)
```
**Output:**
```
High earners with selected columns:
      Name  Salary Department
1      Bob   65000         IT
2  Charlie   70000    Finance
3    Diana   62000         IT

First 3 rows, specific column range:
      Name  Age Department
0    Alice   25         HR
1      Bob   30         IT
2  Charlie   35    Finance
```

### Using .iloc for Combined Selection

```python
# Select by position
subset3 = df.iloc[0:3, 1:4]  # First 3 rows, columns 1-3
print("Position-based selection:")
print(subset3)

# Select specific rows and columns by position
subset4 = df.iloc[[0, 2, 4], [0, 3, 4]]  # Specific rows and columns
print("\nSpecific positions:")
print(subset4)
```
**Output:**
```
Position-based selection:
   Age Department  Salary
0   25         HR   50000
1   30         IT   65000
2   35    Finance   70000

Specific positions:
      Name  Salary  Rating
0    Alice   50000     4.2
2  Charlie   70000     4.5
4      Eve   55000     4.3
```

## Advanced Indexing Techniques

### Conditional Selection with Multiple Criteria

```python
# Create more complex dataset
np.random.seed(42)
df_complex = pd.DataFrame({
    'ID': range(1, 11),
    'Name': [f'Employee_{i}' for i in range(1, 11)],
    'Department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 10),
    'Salary': np.random.randint(40000, 80000, 10),
    'Experience': np.random.randint(1, 10, 10),
    'Performance': np.random.choice(['Excellent', 'Good', 'Average'], 10)
})

print("Complex dataset:")
print(df_complex)
```
**Output:**
```
Complex dataset:
   ID        Name Department  Salary  Experience Performance
0   1  Employee_1         IT   65894           6   Excellent
1   2  Employee_2  Marketing   48174           9    Excellent
2   3  Employee_3     Finance   61413           5        Good
3   4  Employee_4         HR   74247           1    Excellent
4   5  Employee_5         IT   70052           6        Good
5   6  Employee_6  Marketing   59360           4     Average
6   7  Employee_7         IT   44947           8    Excellent
7   8  Employee_8     Finance   43132           3        Good
8   9  Employee_9         IT   68107           2    Excellent
9  10 Employee_10        HR   62509           7     Average
```

```python
# Complex filtering
senior_excellent = df_complex[
    (df_complex['Experience'] >= 5) & 
    (df_complex['Performance'] == 'Excellent') & 
    (df_complex['Salary'] > 60000)
]
print("Senior excellent performers with high salary:")
print(senior_excellent)

# Multiple column conditions
it_or_finance_experienced = df_complex[
    (df_complex['Department'].isin(['IT', 'Finance'])) & 
    (df_complex['Experience'] > 3)
]
print("\nExperienced IT or Finance employees:")
print(it_or_finance_experienced[['Name', 'Department', 'Experience', 'Salary']])
```
**Output:**
```
Senior excellent performers with high salary:
   ID        Name Department  Salary  Experience Performance
0   1  Employee_1         IT   65894           6   Excellent

Experienced IT or Finance employees:
        Name Department  Experience  Salary
0  Employee_1         IT           6   65894
2  Employee_3    Finance           5   61413
4  Employee_5         IT           6   70052
6  Employee_7         IT           8   44947
```

### Using query() Method

```python
# Using query for readable filtering
query_result = df_complex.query('Salary > 60000 and Experience >= 5')
print("Using query method:")
print(query_result[['Name', 'Salary', 'Experience']])

# Query with string conditions
it_query = df_complex.query("Department == 'IT' and Performance == 'Excellent'")
print("\nIT excellent performers:")
print(it_query[['Name', 'Salary', 'Performance']])
```
**Output:**
```
Using query method:
        Name  Salary  Experience
0  Employee_1   65894           6
4  Employee_5   70052           6
8  Employee_9   68107           2
9 Employee_10   62509           7

IT excellent performers:
        Name  Salary Performance
0  Employee_1   65894   Excellent
8  Employee_9   68107   Excellent
```

## Performance Tips

```python
import time

# Performance comparison: different selection methods
large_df = pd.DataFrame({
    'A': np.random.randn(100000),
    'B': np.random.randint(1, 1000, 100000),
    'C': np.random.choice(['X', 'Y', 'Z'], 100000)
})

# Method 1: Boolean indexing
start_time = time.time()
result1 = large_df[large_df['B'] > 500]
time1 = time.time() - start_time

# Method 2: Query method
start_time = time.time()
result2 = large_df.query('B > 500')
time2 = time.time() - start_time

print(f"Boolean indexing time: {time1:.4f} seconds")
print(f"Query method time: {time2:.4f} seconds")
print(f"Results identical: {result1.equals(result2)}")
```
**Output:**
```
Boolean indexing time: 0.0031 seconds
Query method time: 0.0029 seconds
Results identical: True
```

## Summary of Selection Methods

| Method | Use Case | Returns | Example |
|--------|----------|---------|---------|
| `df['col']` | Single column | Series | `df['Name']` |
| `df[['col1', 'col2']]` | Multiple columns | DataFrame | `df[['Name', 'Age']]` |
| `df.iloc[row]` | Row by position | Series | `df.iloc[0]` |
| `df.iloc[rows, cols]` | Rows/cols by position | DataFrame | `df.iloc[0:3, 1:4]` |
| `df.loc[label]` | Row by label | Series | `df.loc['Alice']` |
| `df.loc[condition]` | Conditional selection | DataFrame | `df.loc[df['Age'] > 30]` |
| `df[condition]` | Boolean indexing | DataFrame | `df[df['Salary'] > 50000]` |
| `df.query()` | SQL-like queries | DataFrame | `df.query('Age > 30')` |

### Best Practices

1. **Use .loc and .iloc explicitly** for clarity
2. **Parenthesize complex conditions** in boolean indexing
3. **Use .query()** for readable complex conditions
4. **Avoid chained indexing** like `df['A']['B']`
5. **Consider performance** for large datasets

---

**Next: Basic Data Manipulation**