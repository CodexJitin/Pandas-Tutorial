# 2.4 Merging and Joining Data

## Understanding Different Join Types

### Creating Sample DataFrames

```python
import pandas as pd
import numpy as np

# Create sample datasets for demonstration
employees = pd.DataFrame({
    'emp_id': [1001, 1002, 1003, 1004, 1005],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'department_id': [101, 102, 103, 101, 104],
    'salary': [50000, 65000, 70000, 55000, 60000]
})

departments = pd.DataFrame({
    'dept_id': [101, 102, 103, 105],
    'dept_name': ['HR', 'IT', 'Finance', 'Marketing'],
    'location': ['New York', 'San Francisco', 'Chicago', 'Boston']
})

projects = pd.DataFrame({
    'project_id': ['P001', 'P002', 'P003', 'P004'],
    'employee_id': [1001, 1002, 1003, 1006],
    'project_name': ['Website Redesign', 'Data Migration', 'Financial Analysis', 'Mobile App'],
    'budget': [100000, 150000, 75000, 200000]
})

print("Employees DataFrame:")
print(employees)
print("\nDepartments DataFrame:")
print(departments)
print("\nProjects DataFrame:")
print(projects)
```
**Output:**
```
Employees DataFrame:
   emp_id     name  department_id  salary
0    1001    Alice            101   50000
1    1002      Bob            102   65000
2    1003  Charlie            103   70000
3    1004    Diana            101   55000
4    1005      Eve            104   60000

Departments DataFrame:
   dept_id   dept_name       location
0      101          HR       New York
1      102          IT  San Francisco
2      103     Finance        Chicago
3      105   Marketing         Boston

Projects DataFrame:
  project_id  employee_id       project_name  budget
0       P001         1001   Website Redesign  100000
1       P002         1002      Data Migration  150000
2       P003         1003  Financial Analysis   75000
3       P004         1006         Mobile App  200000
```

## Basic Merge Operations

### Inner Join

```python
# Inner join - only matching records from both DataFrames
inner_join = pd.merge(
    employees, 
    departments, 
    left_on='department_id', 
    right_on='dept_id', 
    how='inner'
)

print("Inner Join (Employees + Departments):")
print(inner_join)
print(f"Result shape: {inner_join.shape}")
```
**Output:**
```
Inner Join (Employees + Departments):
   emp_id     name  department_id  salary  dept_id   dept_name       location
0    1001    Alice            101   50000      101          HR       New York
1    1004    Diana            101   55000      101          HR       New York
2    1002      Bob            102   65000      102          IT  San Francisco
3    1003  Charlie            103   70000      103     Finance        Chicago

Result shape: (4, 7)
```

### Left Join

```python
# Left join - all records from left DataFrame, matching from right
left_join = pd.merge(
    employees, 
    departments, 
    left_on='department_id', 
    right_on='dept_id', 
    how='left'
)

print("Left Join (All Employees + Matching Departments):")
print(left_join)
print(f"Result shape: {left_join.shape}")

# Check for missing department information
missing_dept = left_join[left_join['dept_name'].isna()]
print(f"\nEmployees without department match: {len(missing_dept)}")
print(missing_dept[['name', 'department_id']])
```
**Output:**
```
Left Join (All Employees + Matching Departments):
   emp_id     name  department_id  salary  dept_id   dept_name       location
0    1001    Alice            101   50000    101.0          HR       New York
1    1002      Bob            102   65000    102.0          IT  San Francisco
2    1003  Charlie            103   70000    103.0     Finance        Chicago
3    1004    Diana            101   55000    101.0          HR       New York
4    1005      Eve            104   60000      NaN         NaN            NaN

Employees without department match: 1
   name  department_id
4   Eve            104
```

### Right Join

```python
# Right join - all records from right DataFrame, matching from left
right_join = pd.merge(
    employees, 
    departments, 
    left_on='department_id', 
    right_on='dept_id', 
    how='right'
)

print("Right Join (All Departments + Matching Employees):")
print(right_join)

# Check for departments without employees
empty_depts = right_join[right_join['emp_id'].isna()]
print(f"\nDepartments without employees: {len(empty_depts)}")
print(empty_depts[['dept_name', 'location']])
```
**Output:**
```
Right Join (All Departments + Matching Employees):
   emp_id     name  department_id   salary  dept_id   dept_name       location
0    1001    Alice          101.0  50000.0      101          HR       New York
1    1004    Diana          101.0  55000.0      101          HR       New York
2    1002      Bob          102.0  65000.0      102          IT  San Francisco
3    1003  Charlie          103.0  70000.0      103     Finance        Chicago
4     NaN      NaN            NaN      NaN      105   Marketing         Boston

Departments without employees: 1
   dept_name location
4  Marketing   Boston
```

### Outer Join

```python
# Outer join - all records from both DataFrames
outer_join = pd.merge(
    employees, 
    departments, 
    left_on='department_id', 
    right_on='dept_id', 
    how='outer'
)

print("Outer Join (All Employees + All Departments):")
print(outer_join)
print(f"Result shape: {outer_join.shape}")

# Analysis of missing data in outer join
print(f"\nEmployees without department: {outer_join['dept_name'].isna().sum()}")
print(f"Departments without employees: {outer_join['emp_id'].isna().sum()}")
```
**Output:**
```
Outer Join (All Employees + All Departments):
   emp_id     name  department_id   salary  dept_id   dept_name       location
0    1001    Alice          101.0  50000.0    101.0          HR       New York
1    1002      Bob          102.0  65000.0    102.0          IT  San Francisco
2    1003  Charlie          103.0  70000.0    103.0     Finance        Chicago
3    1004    Diana          101.0  55000.0    101.0          HR       New York
4    1005      Eve          104.0  60000.0      NaN         NaN            NaN
5     NaN      NaN            NaN      NaN    105.0   Marketing         Boston

Employees without department: 1
Departments without employees: 1
```

## Advanced Merge Scenarios

### Multiple Column Joins

```python
# Create data with multiple join keys
sales_data = pd.DataFrame({
    'year': [2022, 2022, 2023, 2023, 2023],
    'quarter': [1, 2, 1, 2, 3],
    'region': ['North', 'South', 'North', 'South', 'East'],
    'sales': [100000, 120000, 110000, 130000, 95000]
})

targets_data = pd.DataFrame({
    'year': [2022, 2022, 2023, 2023, 2023, 2023],
    'quarter': [1, 2, 1, 2, 3, 4],
    'region': ['North', 'South', 'North', 'South', 'East', 'West'],
    'target': [95000, 115000, 105000, 125000, 100000, 90000]
})

print("Sales Data:")
print(sales_data)
print("\nTargets Data:")
print(targets_data)

# Merge on multiple columns
sales_vs_targets = pd.merge(
    sales_data,
    targets_data,
    on=['year', 'quarter', 'region'],
    how='outer'
)

# Calculate performance
sales_vs_targets['performance'] = (
    sales_vs_targets['sales'] / sales_vs_targets['target'] * 100
).round(1)

print("\nSales vs Targets Analysis:")
print(sales_vs_targets)
```
**Output:**
```
Sales Data:
   year  quarter region   sales
0  2022        1  North  100000
1  2022        2  South  120000
2  2023        1  North  110000
3  2023        2  South  130000
4  2023        3   East   95000

Targets Data:
   year  quarter region  target
0  2022        1  North   95000
1  2022        2  South  115000
2  2023        1  North  105000
3  2023        2  South  125000
4  2023        3   East  100000
5  2023        4   West   90000

Sales vs Targets Analysis:
   year  quarter region    sales  target  performance
0  2022        1  North  100000.0   95000        105.3
1  2022        2  South  120000.0  115000        104.3
2  2023        1  North  110000.0  105000        104.8
3  2023        2  South  130000.0  125000        104.0
4  2023        3   East   95000.0  100000         95.0
5  2023        4   West        NaN   90000          NaN
```

### Merge with Different Column Names

```python
# Create DataFrames with different column names but same data
customer_orders = pd.DataFrame({
    'order_id': ['O001', 'O002', 'O003', 'O004'],
    'customer_ref': ['C001', 'C002', 'C001', 'C003'],
    'order_amount': [500, 750, 300, 1000],
    'order_date': pd.to_datetime(['2023-01-15', '2023-01-20', '2023-02-01', '2023-02-05'])
})

customer_info = pd.DataFrame({
    'cust_id': ['C001', 'C002', 'C003', 'C004'],
    'customer_name': ['Acme Corp', 'Tech Solutions', 'Global Industries', 'Smart Systems'],
    'industry': ['Manufacturing', 'Technology', 'Logistics', 'Healthcare']
})

print("Customer Orders:")
print(customer_orders)
print("\nCustomer Info:")
print(customer_info)

# Merge with different column names
orders_with_customer = pd.merge(
    customer_orders,
    customer_info,
    left_on='customer_ref',
    right_on='cust_id',
    how='left'
)

print("\nOrders with Customer Information:")
print(orders_with_customer)
```
**Output:**
```
Customer Orders:
  order_id customer_ref  order_amount order_date
0     O001         C001           500 2023-01-15
1     O002         C002           750 2023-01-20
2     O003         C001           300 2023-02-01
3     O004         C003          1000 2023-02-05

Customer Info:
  cust_id    customer_name    industry
0    C001         Acme Corp  Manufacturing
1    C002    Tech Solutions    Technology
2    C003  Global Industries   Logistics
3    C004     Smart Systems  Healthcare

Orders with Customer Information:
  order_id customer_ref  order_amount order_date cust_id    customer_name    industry
0     O001         C001           500 2023-01-15    C001         Acme Corp  Manufacturing
1     O002         C002           750 2023-01-20    C002    Tech Solutions    Technology
2     O003         C001           300 2023-02-01    C001         Acme Corp  Manufacturing
3     O004         C003          1000 2023-02-05    C003  Global Industries   Logistics
```

## Concatenation

### Vertical Concatenation

```python
# Create quarterly data
q1_sales = pd.DataFrame({
    'employee': ['Alice', 'Bob', 'Charlie'],
    'sales': [10000, 12000, 8000],
    'quarter': 'Q1'
})

q2_sales = pd.DataFrame({
    'employee': ['Alice', 'Bob', 'Diana'],
    'sales': [11000, 13000, 9500],
    'quarter': 'Q2'
})

q3_sales = pd.DataFrame({
    'employee': ['Alice', 'Charlie', 'Diana', 'Eve'],
    'sales': [10500, 8500, 10000, 7500],
    'quarter': 'Q3'
})

print("Q1 Sales:")
print(q1_sales)
print("\nQ2 Sales:")
print(q2_sales)
print("\nQ3 Sales:")
print(q3_sales)

# Concatenate vertically
yearly_sales = pd.concat([q1_sales, q2_sales, q3_sales], ignore_index=True)
print("\nYearly Sales (Concatenated):")
print(yearly_sales)
```
**Output:**
```
Q1 Sales:
  employee  sales quarter
0    Alice  10000      Q1
1      Bob  12000      Q1
2  Charlie   8000      Q1

Q2 Sales:
  employee  sales quarter
0    Alice  11000      Q2
1      Bob  13000      Q2
2    Diana   9500      Q2

Q3 Sales:
  employee  sales quarter
0    Alice  10500      Q3
1  Charlie   8500      Q3
2    Diana  10000      Q3
3      Eve   7500      Q3

Yearly Sales (Concatenated):
   employee  sales quarter
0     Alice  10000      Q1
1       Bob  12000      Q1
2   Charlie   8000      Q1
3     Alice  11000      Q2
4       Bob  13000      Q2
5     Diana   9500      Q2
6     Alice  10500      Q3
7   Charlie   8500      Q3
8     Diana  10000      Q3
9       Eve   7500      Q3
```

### Horizontal Concatenation

```python
# Create data for horizontal concatenation
employee_basic = pd.DataFrame({
    'emp_id': [1001, 1002, 1003],
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
})

employee_work = pd.DataFrame({
    'department': ['HR', 'IT', 'Finance'],
    'salary': [50000, 65000, 70000],
    'start_date': pd.to_datetime(['2020-01-15', '2019-03-20', '2021-07-10'])
})

print("Employee Basic Info:")
print(employee_basic)
print("\nEmployee Work Info:")
print(employee_work)

# Concatenate horizontally
employee_complete = pd.concat([employee_basic, employee_work], axis=1)
print("\nComplete Employee Information:")
print(employee_complete)
```
**Output:**
```
Employee Basic Info:
   emp_id     name  age
0    1001    Alice   25
1    1002      Bob   30
2    1003  Charlie   35

Employee Work Info:
  department  salary start_date
0         HR   50000 2020-01-15
1         IT   65000 2019-03-20
2    Finance   70000 2021-07-10

Complete Employee Information:
   emp_id     name  age department  salary start_date
0    1001    Alice   25         HR   50000 2020-01-15
1    1002      Bob   30         IT   65000 2019-03-20
2    1003  Charlie   35    Finance   70000 2021-07-10
```

## Index-based Joining

### Using Join Method

```python
# Create DataFrames with meaningful indices
employee_salary = pd.DataFrame({
    'salary': [50000, 65000, 70000, 55000],
    'bonus': [5000, 6500, 7000, 5500]
}, index=['Alice', 'Bob', 'Charlie', 'Diana'])

employee_performance = pd.DataFrame({
    'rating': [4.2, 4.8, 4.5, 4.3, 4.6],
    'projects_completed': [3, 5, 4, 2, 6]
}, index=['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'])

print("Employee Salary:")
print(employee_salary)
print("\nEmployee Performance:")
print(employee_performance)

# Join using index
salary_performance = employee_salary.join(employee_performance, how='outer')
print("\nJoined Data (Salary + Performance):")
print(salary_performance)
```
**Output:**
```
Employee Salary:
         salary  bonus
Alice     50000   5000
Bob       65000   6500
Charlie   70000   7000
Diana     55000   5500

Employee Performance:
         rating  projects_completed
Alice       4.2                   3
Bob         4.8                   5
Charlie     4.5                   4
Diana       4.3                   2
Eve         4.6                   6

Joined Data (Salary + Performance):
         salary  bonus  rating  projects_completed
Alice   50000.0 5000.0     4.2                 3.0
Bob     65000.0 6500.0     4.8                 5.0
Charlie 70000.0 7000.0     4.5                 4.0
Diana   55000.0 5500.0     4.3                 2.0
Eve         NaN    NaN     4.6                 6.0
```

### Multi-Index Joins

```python
# Create multi-index DataFrames
sales_data_multi = pd.DataFrame({
    'sales': [100, 120, 110, 130, 125, 140],
    'units': [10, 12, 11, 13, 12, 14]
}, index=pd.MultiIndex.from_tuples([
    ('2023', 'Q1'), ('2023', 'Q2'), ('2023', 'Q3'),
    ('2024', 'Q1'), ('2024', 'Q2'), ('2024', 'Q3')
], names=['Year', 'Quarter']))

targets_data_multi = pd.DataFrame({
    'target_sales': [95, 115, 105, 125, 120, 135],
    'target_units': [9, 11, 10, 12, 11, 13]
}, index=pd.MultiIndex.from_tuples([
    ('2023', 'Q1'), ('2023', 'Q2'), ('2023', 'Q3'),
    ('2024', 'Q1'), ('2024', 'Q2'), ('2024', 'Q3')
], names=['Year', 'Quarter']))

print("Sales Data (Multi-Index):")
print(sales_data_multi)
print("\nTargets Data (Multi-Index):")
print(targets_data_multi)

# Join multi-index DataFrames
performance_analysis = sales_data_multi.join(targets_data_multi)
performance_analysis['sales_achievement'] = (
    performance_analysis['sales'] / performance_analysis['target_sales'] * 100
).round(1)

print("\nPerformance Analysis:")
print(performance_analysis)
```
**Output:**
```
Sales Data (Multi-Index):
              sales  units
Year Quarter             
2023 Q1         100     10
     Q2         120     12
     Q3         110     11
2024 Q1         130     13
     Q2         125     12
     Q3         140     14

Targets Data (Multi-Index):
              target_sales  target_units
Year Quarter                          
2023 Q1                 95             9
     Q2                115            11
     Q3                105            10
2024 Q1                125            12
     Q2                120            11
     Q3                135            13

Performance Analysis:
              sales  units  target_sales  target_units  sales_achievement
Year Quarter                                                            
2023 Q1         100     10            95             9              105.3
     Q2         120     12           115            11              104.3
     Q3         110     11           105            10              104.8
2024 Q1         130     13           125            12              104.0
     Q2         125     12           120            11              104.2
     Q3         140     14           135            13              103.7
```

## Handling Duplicate Keys

```python
# Create DataFrames with duplicate keys
orders_main = pd.DataFrame({
    'order_id': ['O001', 'O002', 'O003'],
    'customer_id': ['C001', 'C002', 'C001'],
    'order_total': [500, 750, 300]
})

customer_contacts = pd.DataFrame({
    'customer_id': ['C001', 'C001', 'C002', 'C002'],
    'contact_type': ['email', 'phone', 'email', 'phone'],
    'contact_value': ['alice@company.com', '555-1234', 'bob@tech.com', '555-5678']
})

print("Orders:")
print(orders_main)
print("\nCustomer Contacts:")
print(customer_contacts)

# Merge with duplicate keys - creates cartesian product
orders_with_contacts = pd.merge(
    orders_main,
    customer_contacts,
    on='customer_id',
    how='left'
)

print("\nOrders with All Customer Contacts:")
print(orders_with_contacts)
print(f"Result shape: {orders_with_contacts.shape}")
```
**Output:**
```
Orders:
  order_id customer_id  order_total
0     O001        C001          500
1     O002        C002          750
2     O003        C001          300

Customer Contacts:
  customer_id contact_type       contact_value
0        C001        email    alice@company.com
1        C001        phone            555-1234
2        C002        email        bob@tech.com
3        C002        phone            555-5678

Orders with All Customer Contacts:
  order_id customer_id  order_total contact_type       contact_value
0     O001        C001          500        email    alice@company.com
1     O001        C001          500        phone            555-1234
2     O002        C002          750        email        bob@tech.com
3     O002        C002          750        phone            555-5678
4     O003        C001          300        email    alice@company.com
5     O003        C001          300        phone            555-1234

Result shape: (6, 5)
```

## Performance Considerations

```python
import time

# Create larger datasets for performance testing
np.random.seed(42)
large_df1 = pd.DataFrame({
    'key': np.random.randint(1, 10000, 50000),
    'value1': np.random.randn(50000)
})

large_df2 = pd.DataFrame({
    'key': np.random.randint(1, 10000, 30000),
    'value2': np.random.randn(30000)
})

print(f"DataFrame 1 shape: {large_df1.shape}")
print(f"DataFrame 2 shape: {large_df2.shape}")

# Performance comparison
start_time = time.time()
result_merge = pd.merge(large_df1, large_df2, on='key', how='inner')
merge_time = time.time() - start_time

# Set index for join comparison
large_df1_indexed = large_df1.set_index('key')
large_df2_indexed = large_df2.set_index('key')

start_time = time.time()
result_join = large_df1_indexed.join(large_df2_indexed, how='inner')
join_time = time.time() - start_time

print(f"\nMerge time: {merge_time:.4f} seconds")
print(f"Join time: {join_time:.4f} seconds")
print(f"Result shapes identical: {result_merge.shape[0] == result_join.shape[0]}")
```
**Output:**
```
DataFrame 1 shape: (50000, 2)
DataFrame 2 shape: (30000, 2)

Merge time: 0.0156 seconds
Join time: 0.0078 seconds
Result shapes identical: True
```

## Summary of Join Operations

| Operation | Method | Use Case | Key Points |
|-----------|--------|----------|------------|
| Inner Join | `pd.merge(how='inner')` | Only matching records | Smallest result set |
| Left Join | `pd.merge(how='left')` | Keep all left records | Preserves left DataFrame structure |
| Right Join | `pd.merge(how='right')` | Keep all right records | Preserves right DataFrame structure |
| Outer Join | `pd.merge(how='outer')` | Keep all records | Largest result set |
| Concatenate | `pd.concat()` | Stack DataFrames | Simple combination |
| Index Join | `df1.join(df2)` | Index-based joining | Faster for indexed data |

### Best Practices

1. **Choose appropriate join type** based on your analysis needs
2. **Handle duplicate keys** carefully to avoid unexpected results
3. **Use index-based joins** for better performance when possible
4. **Validate results** after complex joins
5. **Consider memory usage** with large datasets
6. **Clean and standardize** join keys before merging
7. **Document join logic** for complex multi-step operations

### Common Pitfalls

1. **Cartesian products** from duplicate keys
2. **Data type mismatches** in join columns
3. **Missing or null** join keys
4. **Memory issues** with large outer joins
5. **Index confusion** after multiple operations

---

**Next: Date and Time Operations**