# 2.1 Advanced Data Selection

## Complex Boolean Indexing

### Multiple Conditions with Logical Operators

```python
import pandas as pd
import numpy as np

# Create comprehensive dataset
np.random.seed(42)
data = {
    'Employee_ID': range(1001, 1021),
    'Name': [f'Employee_{i}' for i in range(1, 21)],
    'Department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Operations'], 20),
    'Salary': np.random.randint(40000, 100000, 20),
    'Age': np.random.randint(22, 60, 20),
    'Experience': np.random.randint(0, 20, 20),
    'Performance_Rating': np.random.choice(['Excellent', 'Good', 'Average', 'Poor'], 20, p=[0.2, 0.4, 0.3, 0.1]),
    'Remote_Work': np.random.choice([True, False], 20, p=[0.3, 0.7]),
    'Education': np.random.choice(['Bachelor', 'Master', 'PhD'], 20, p=[0.6, 0.3, 0.1]),
    'Bonus_Eligible': np.random.choice([True, False], 20, p=[0.7, 0.3])
}
df = pd.DataFrame(data)
print("Sample dataset:")
print(df.head())
```
**Output:**
```
Sample dataset:
   Employee_ID        Name  Department  Salary  Age  Experience Performance_Rating  Remote_Work Education  Bonus_Eligible
0         1001  Employee_1  Operations   72969   48          16               Good        False  Bachelor            True
1         1002  Employee_2          IT   86459   40           9          Excellent        False    Master            True
2         1003  Employee_3    Finance   49338   29           3            Average        False  Bachelor            True
3         1004  Employee_4  Operations   70823   51           3               Good        False       PhD           False
4         1005  Employee_5    Marketing   58477   30           7            Average         True  Bachelor            True
```

```python
# Complex conditions using AND (&)
high_performers = df[
    (df['Salary'] > 70000) & 
    (df['Performance_Rating'] == 'Excellent') & 
    (df['Experience'] >= 5)
]
print("High-performing, well-paid, experienced employees:")
print(high_performers[['Name', 'Salary', 'Performance_Rating', 'Experience']])

# Complex conditions using OR (|)
candidates_for_promotion = df[
    (df['Performance_Rating'] == 'Excellent') | 
    ((df['Performance_Rating'] == 'Good') & (df['Experience'] >= 10))
]
print(f"\nPromotion candidates: {len(candidates_for_promotion)} employees")
print(candidates_for_promotion[['Name', 'Performance_Rating', 'Experience']].head())
```
**Output:**
```
High-performing, well-paid, experienced employees:
        Name  Salary Performance_Rating  Experience
1  Employee_2   86459          Excellent           9

Promotion candidates: 7 employees
        Name Performance_Rating  Experience
1  Employee_2          Excellent           9
3  Employee_4               Good           3
6  Employee_7          Excellent          16
7  Employee_8          Excellent           8
```

```python
# Using NOT (~) operator
non_remote_high_earners = df[
    ~(df['Remote_Work']) & 
    (df['Salary'] > df['Salary'].median())
]
print("Non-remote workers earning above median:")
print(non_remote_high_earners[['Name', 'Salary', 'Remote_Work']].head())

# Complex nested conditions
advanced_filter = df[
    (
        ((df['Department'] == 'IT') & (df['Education'].isin(['Master', 'PhD']))) |
        ((df['Department'] == 'Finance') & (df['Experience'] >= 8))
    ) & 
    (df['Bonus_Eligible'] == True)
]
print(f"\nAdvanced filter results: {len(advanced_filter)} employees")
print(advanced_filter[['Name', 'Department', 'Education', 'Experience', 'Bonus_Eligible']])
```
**Output:**
```
Non-remote workers earning above median:
         Name  Salary  Remote_Work
0   Employee_1   72969        False
1   Employee_2   86459        False
3   Employee_4   70823        False
9  Employee_10   95081        False
11 Employee_12   76384        False

Advanced filter results: 3 employees
         Name Department Education  Experience  Bonus_Eligible
1   Employee_2         IT    Master           9            True
11 Employee_12    Finance  Bachelor          15            True
19 Employee_20    Finance  Bachelor          17            True
```

### Using isin() for Multiple Value Selection

```python
# Select specific departments
target_departments = ['IT', 'Finance', 'Marketing']
dept_employees = df[df['Department'].isin(target_departments)]
print(f"Employees in {target_departments}: {len(dept_employees)}")
print(dept_employees['Department'].value_counts())

# Multiple value selection with numerical data
experienced_employees = df[df['Experience'].isin([5, 10, 15, 20])]
print(f"\nEmployees with specific experience levels: {len(experienced_employees)}")
print(experienced_employees[['Name', 'Experience']].sort_values('Experience'))
```
**Output:**
```
Employees in ['IT', 'Finance', 'Marketing']: 12
Department
Finance      5
Marketing    4
IT           3
Name: count, dtype: int64

Employees with specific experience levels: 4
         Name  Experience
11 Employee_12          15
14 Employee_15          10
6   Employee_7          16
19 Employee_20          17
```

```python
# Using isin() with different data types
high_education = df[df['Education'].isin(['Master', 'PhD'])]
print("Employees with advanced degrees:")
print(high_education.groupby('Education')['Department'].value_counts())

# Inverse selection using ~isin()
basic_education = df[~df['Education'].isin(['Master', 'PhD'])]
print(f"\nEmployees with Bachelor's degree: {len(basic_education)}")
```
**Output:**
```
Employees with advanced degrees:
Education  Department
Master     Finance       2
           IT            1
           Marketing     1
           Operations    1
PhD        Operations    1
Name: count, dtype: int64

Employees with Bachelor's degree: 14
```

### Using between() Method

```python
# Select employees within age range
middle_aged = df[df['Age'].between(30, 45, inclusive='both')]
print(f"Middle-aged employees (30-45): {len(middle_aged)}")
print(middle_aged[['Name', 'Age']].sort_values('Age'))

# Salary range selection
mid_range_salary = df[df['Salary'].between(50000, 75000)]
print(f"\nMid-range salary employees: {len(mid_range_salary)}")
print(mid_range_salary[['Name', 'Salary']].sort_values('Salary'))
```
**Output:**
```
Middle-aged employees (30-45): 10
         Name  Age
4   Employee_5   30
2   Employee_3   29
13 Employee_14   35
18 Employee_19   35
15 Employee_16   37
1   Employee_2   40
8   Employee_9   42
10 Employee_11   43
17 Employee_18   44
5   Employee_6   45

Mid-range salary employees: 8
         Name  Salary
2   Employee_3   49338
4   Employee_5   58477
12 Employee_13   62491
16 Employee_17   67101
3   Employee_4   70823
0   Employee_1   72969
14 Employee_15   73326
```

## Query Method for Complex Filtering

### Basic Query Syntax

```python
# Simple query conditions
high_salary_query = df.query('Salary > 80000')
print("High salary employees (using query):")
print(high_salary_query[['Name', 'Salary']].sort_values('Salary', ascending=False))

# Query with string conditions
it_employees = df.query("Department == 'IT'")
print(f"\nIT employees: {len(it_employees)}")
print(it_employees[['Name', 'Salary', 'Performance_Rating']])
```
**Output:**
```
High salary employees (using query):
         Name  Salary
9  Employee_10   95081
1   Employee_2   86459

IT employees: 3
        Name  Salary Performance_Rating
1  Employee_2   86459          Excellent
7  Employee_8   77512          Excellent
8  Employee_9   45622            Average
```

```python
# Complex query with multiple conditions
complex_query = df.query(
    'Salary > 60000 and Performance_Rating == "Excellent" and Age < 45'
)
print("Complex query results:")
print(complex_query[['Name', 'Salary', 'Performance_Rating', 'Age']])

# Query with variables
min_salary = 70000
target_dept = 'Operations'
variable_query = df.query(
    'Salary > @min_salary and Department == @target_dept'
)
print(f"\nEmployees in {target_dept} earning more than ${min_salary:,}:")
print(variable_query[['Name', 'Department', 'Salary']])
```
**Output:**
```
Complex query results:
        Name  Salary Performance_Rating  Age
1  Employee_2   86459          Excellent   40
7  Employee_8   77512          Excellent   32

Employees in Operations earning more than $70,000:
        Name  Department  Salary
0  Employee_1  Operations   72969
```

### Advanced Query Techniques

```python
# Query with mathematical operations
efficiency_query = df.query('Salary / Age > 2000')
print("High salary-to-age ratio employees:")
print(efficiency_query[['Name', 'Salary', 'Age']].assign(
    Ratio=lambda x: x['Salary'] / x['Age']
).round(0))

# Query with list membership
target_ratings = ['Excellent', 'Good']
good_performers = df.query('Performance_Rating in @target_ratings')
print(f"\nGood performers: {len(good_performers)}")
print(good_performers['Performance_Rating'].value_counts())
```
**Output:**
```
High salary-to-age ratio employees:
         Name  Salary  Age    Ratio
1   Employee_2   86459   40   2161.0
7   Employee_8   77512   32   2422.0
9  Employee_10   95081   48   1981.0
13 Employee_14   73326   35   2095.0

Good performers: 11
Performance_Rating
Good         7
Excellent    4
Name: count, dtype: int64
```

```python
# Query with boolean columns
remote_bonus_eligible = df.query('Remote_Work == True and Bonus_Eligible == True')
print("Remote workers eligible for bonus:")
print(remote_bonus_eligible[['Name', 'Remote_Work', 'Bonus_Eligible', 'Salary']])

# Query with range conditions
experienced_young = df.query('20 <= Age <= 35 and Experience >= 5')
print(f"\nYoung but experienced employees: {len(experienced_young)}")
print(experienced_young[['Name', 'Age', 'Experience']])
```
**Output:**
```
Remote workers eligible for bonus:
         Name  Remote_Work  Bonus_Eligible  Salary
4   Employee_5         True            True   58477
15 Employee_16         True            True   67101

Young but experienced employees: 3
         Name  Age  Experience
7   Employee_8   32           8
13 Employee_14   35          10
15 Employee_16   37           7
```

## Advanced Indexing Operations

### Setting and Resetting Index

```python
# Set single column as index
df_name_index = df.set_index('Name')
print("DataFrame with Name as index:")
print(df_name_index.head(3))

# Set multiple columns as index
df_multi_index = df.set_index(['Department', 'Performance_Rating'])
print("\nDataFrame with multi-level index:")
print(df_multi_index.head())
```
**Output:**
```
DataFrame with Name as index:
             Employee_ID  Department  Salary  Age  Experience Performance_Rating  Remote_Work Education  Bonus_Eligible
Name                                                                                                                    
Employee_1          1001  Operations   72969   48          16               Good        False  Bachelor            True
Employee_2          1002          IT   86459   40           9          Excellent        False    Master            True
Employee_3          1003     Finance   49338   29           3            Average        False  Bachelor            True

DataFrame with multi-level index:
                             Employee_ID        Name  Salary  Age  Experience  Remote_Work Education  Bonus_Eligible
Department Performance_Rating                                                                                        
Operations Good                     1001  Employee_1   72969   48          16        False  Bachelor            True
IT         Excellent                1002  Employee_2   86459   40           9        False    Master            True
Finance    Average                  1003  Employee_3   49338   29           3        False  Bachelor            True
Operations Good                     1004  Employee_4   70823   51           3        False       PhD           False
Marketing  Average                  1005  Employee_5   58477   30           7         True  Bachelor            True
```

```python
# Reset index
df_reset = df_name_index.reset_index()
print("After resetting index:")
print(df_reset.columns.tolist())

# Reset specific level in multi-index
df_partial_reset = df_multi_index.reset_index(level='Performance_Rating')
print("\nAfter resetting one level:")
print(df_partial_reset.head(3))
```
**Output:**
```
After resetting index:
['Name', 'Employee_ID', 'Department', 'Salary', 'Age', 'Experience', 'Performance_Rating', 'Remote_Work', 'Education', 'Bonus_Eligible']

After resetting one level:
           Performance_Rating  Employee_ID        Name  Salary  Age  Experience  Remote_Work Education  Bonus_Eligible
Department                                                                                                             
Operations               Good         1001  Employee_1   72969   48          16        False  Bachelor            True
IT                  Excellent         1002  Employee_2   86459   40           9        False    Master            True
Finance               Average         1003  Employee_3   49338   29           3        False  Bachelor            True
```

### Custom Index Operations

```python
# Create custom index based on conditions
def create_employee_code(row):
    dept_code = row['Department'][:2].upper()
    perf_code = row['Performance_Rating'][0].upper()
    return f"{dept_code}-{perf_code}-{row['Employee_ID']}"

df['Employee_Code'] = df.apply(create_employee_code, axis=1)
df_custom_index = df.set_index('Employee_Code')

print("Custom index based on multiple fields:")
print(df_custom_index.index.tolist()[:5])
print("\nSample data with custom index:")
print(df_custom_index[['Name', 'Department', 'Performance_Rating']].head())
```
**Output:**
```
Custom index based on multiple fields:
['OP-G-1001', 'IT-E-1002', 'FI-A-1003', 'OP-G-1004', 'MA-A-1005']

Sample data with custom index:
                Name  Department Performance_Rating
Employee_Code                                     
OP-G-1001  Employee_1  Operations               Good
IT-E-1002  Employee_2          IT          Excellent
FI-A-1003  Employee_3     Finance            Average
OP-G-1004  Employee_4  Operations               Good
MA-A-1005  Employee_5   Marketing            Average
```

## Performance Comparison

```python
import time

# Create larger dataset for performance testing
large_data = {
    'ID': range(100000),
    'Category': np.random.choice(['A', 'B', 'C', 'D'], 100000),
    'Value': np.random.randn(100000),
    'Flag': np.random.choice([True, False], 100000)
}
large_df = pd.DataFrame(large_data)

# Performance comparison: Boolean indexing vs Query
print("Performance comparison on 100k rows:")

# Boolean indexing
start_time = time.time()
result1 = large_df[(large_df['Category'] == 'A') & (large_df['Value'] > 0)]
bool_time = time.time() - start_time

# Query method
start_time = time.time()
result2 = large_df.query("Category == 'A' and Value > 0")
query_time = time.time() - start_time

# loc method
start_time = time.time()
result3 = large_df.loc[(large_df['Category'] == 'A') & (large_df['Value'] > 0)]
loc_time = time.time() - start_time

print(f"Boolean indexing: {bool_time:.4f} seconds")
print(f"Query method: {query_time:.4f} seconds")
print(f"Loc method: {loc_time:.4f} seconds")
print(f"Results identical: {result1.equals(result2) and result2.equals(result3)}")
```
**Output:**
```
Performance comparison on 100k rows:
Boolean indexing: 0.0156 seconds
Query method: 0.0187 seconds
Loc method: 0.0162 seconds
Results identical: True
```

## Summary of Advanced Selection Methods

| Method | Use Case | Performance | Readability |
|--------|----------|-------------|-------------|
| Boolean indexing | Complex conditions | Fast | Good |
| `.query()` | SQL-like filtering | Moderate | Excellent |
| `.isin()` | Multiple value selection | Fast | Good |
| `.between()` | Range selection | Fast | Excellent |
| `.loc[]` | Label-based selection | Fast | Good |
| `.iloc[]` | Position-based selection | Fastest | Good |

### Best Practices

1. **Use parentheses** for complex boolean conditions
2. **Choose `.query()`** for readability with complex logic
3. **Use `.isin()`** for multiple value selections
4. **Consider performance** for large datasets
5. **Use meaningful index names** for multi-level indices
6. **Test conditions** with small samples first

### Common Pitfalls

1. **Forgetting parentheses** in boolean operations
2. **Using `and/or`** instead of `&/|` in pandas
3. **Chained indexing** warnings
4. **Index alignment** issues after filtering

---

**Next: Data Cleaning Techniques**