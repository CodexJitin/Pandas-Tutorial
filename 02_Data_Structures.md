# 1.2 Data Structures - Series and DataFrames

## Series

A Series is a one-dimensional labeled array capable of holding any data type.

### Creating Series

```python
import pandas as pd
import numpy as np

# From a list
s1 = pd.Series([1, 2, 3, 4, 5])
print(s1)
```
```
0    1
1    2
2    3
3    4
4    5
dtype: int64
```

```python
# From a list with custom index
s2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s2)
```
```
a    10
b    20
c    30
dtype: int64
```

```python
# From a dictionary
data_dict = {'Apple': 100, 'Banana': 150, 'Orange': 80}
s3 = pd.Series(data_dict)
print(s3)
```
```
Apple     100
Banana    150
Orange     80
dtype: int64
```

```python
# From NumPy array
arr = np.array([1, 2, 3, 4])
s4 = pd.Series(arr, index=['w', 'x', 'y', 'z'])
print(s4)
```
```
w    1
x    2
y    3
z    4
dtype: int32
```

```python
# From scalar value
s5 = pd.Series(5, index=['a', 'b', 'c', 'd'])
print(s5)
```
```
a    5
b    5
c    5
d    5
dtype: int64
```

### Series Properties

```python
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])

# Basic properties
print(f"Values: {s.values}")      # NumPy array of values
print(f"Index: {s.index}")        # Index object
print(f"Shape: {s.shape}")        # Dimensions
print(f"Size: {s.size}")          # Number of elements
print(f"Data type: {s.dtype}")    # Data type
print(f"Name: {s.name}")          # Series name (None by default)
```
```
Values: [10 20 30 40]
Index: Index(['a', 'b', 'c', 'd'], dtype='object')
Shape: (4,)
Size: 4
Data type: int64
Name: None
```

```python
# Setting name
s.name = "Sample Series"
s.index.name = "Letters"
print(s)
```
```
Letters
a    10
b    20
c    30
d    40
Name: Sample Series, dtype: int64
```

### Indexing and Accessing Data

```python
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])

# Label-based indexing
print(s['a'])           # Single element
print(s[['a', 'c']])    # Multiple elements
```
```
10
a    10
c    30
dtype: int64
```

```python
# Position-based indexing
print(s[0])             # First element
print(s[1:3])           # Slice
```
```
10
b    20
c    30
dtype: int64
```

```python
# Boolean indexing
print(s[s > 25])        # Elements greater than 25
```
```
c    30
d    40
dtype: int64
```
c    30
d    40
dtype: int64
```

```python
# Using .loc and .iloc
print(s.loc['b'])       # Label-based
print(s.iloc[1])        # Position-based
```
```
20
20
```

### Basic Operations on Series

```python
s = pd.Series([1, 2, 3, 4, 5])

# Arithmetic operations
print(s + 10)           # Add scalar
print(s * 2)            # Multiply by scalar
print(s ** 2)           # Power
```
```
0    11
1    12
2    13
3    14
4    15
dtype: int64
0     2
1     4
2     6
3     8
4    10
dtype: int64
0     1
1     4
2     9
3    16
4    25
dtype: int64
```

```python
# Statistical operations
print(s.sum())          # Sum
print(s.mean())         # Mean
print(s.std())          # Standard deviation
print(s.describe())     # Descriptive statistics
```
```
15
3.0
1.5811388300841898
count    5.000000
mean     3.000000
std      1.581139
min      1.000000
25%      2.000000
50%      3.000000
75%      4.000000
max      5.000000
dtype: float64
```

## DataFrames

A DataFrame is a two-dimensional labeled data structure with columns of potentially different types.

### Creating DataFrames

```python
# From dictionary of lists
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'London', 'Paris', 'Tokyo'],
    'Salary': [50000, 60000, 70000, 55000]
}
df1 = pd.DataFrame(data)
print(df1)
```
```
      Name  Age      City  Salary
0    Alice   25  New York   50000
1      Bob   30    London   60000
2  Charlie   35     Paris   70000
3    Diana   28     Tokyo   55000
```

```python
# From list of dictionaries
data_list = [
    {'Name': 'Alice', 'Age': 25, 'City': 'New York'},
    {'Name': 'Bob', 'Age': 30, 'City': 'London'},
    {'Name': 'Charlie', 'Age': 35, 'City': 'Paris'}
]
df2 = pd.DataFrame(data_list)
print(df2)
```
```
      Name  Age      City
0    Alice   25  New York
1      Bob   30    London
2  Charlie   35     Paris
```

```python
# From NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df3 = pd.DataFrame(arr, columns=['A', 'B', 'C'], index=['X', 'Y', 'Z'])
print(df3)
```
```
   A  B  C
X  1  2  3
Y  4  5  6
Z  7  8  9
```

```python
# From Series
s1 = pd.Series([1, 2, 3], name='Column1')
s2 = pd.Series([4, 5, 6], name='Column2')
df4 = pd.DataFrame([s1, s2]).T  # Transpose to get correct orientation
print(df4)
```
```
   Column1  Column2
0        1        4
1        2        5
2        3        6
```

### DataFrame Structure and Properties

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': ['x', 'y', 'z', 'w']
})

# Basic properties
print(f"Shape: {df.shape}")         # (rows, columns)
print(f"Size: {df.size}")           # Total number of elements
print(f"Dimensions: {df.ndim}")     # Number of dimensions
print(f"Columns: {df.columns}")     # Column names
print(f"Index: {df.index}")         # Row index
print(f"Data types:\n{df.dtypes}")  # Data types of each column
```
```
Shape: (4, 3)
Size: 12
Dimensions: 2
Columns: Index(['A', 'B', 'C'], dtype='object')
Index: RangeIndex(start=0, stop=4, step=1)
Data types:
A      int64
B      int64
C     object
dtype: object
```

```python
# More detailed information
print(df.info())                    # Comprehensive info
print(df.describe())                # Statistical summary
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   A       4 non-null      int64 
 1   B       4 non-null      int64 
 2   C       4 non-null      object
dtypes: int64(2), object(1)
memory usage: 224.0+ bytes
None
              A         B
count  4.000000  4.000000
mean   2.500000  6.500000
std    1.290994  1.290994
min    1.000000  5.000000
25%    1.750000  5.750000
50%    2.500000  6.500000
75%    3.250000  7.250000
max    4.000000  8.000000
```

### Understanding Index and Columns

```python
# Creating DataFrame with custom index
df = pd.DataFrame({
    'Score': [85, 92, 78, 96],
    'Grade': ['B', 'A', 'C', 'A']
}, index=['Alice', 'Bob', 'Charlie', 'Diana'])

print("Original DataFrame:")
print(df)
```
```
Original DataFrame:
         Score Grade
Alice       85     B
Bob         92     A
Charlie     78     C
Diana       96     A
```

```python
# Accessing index and columns
print(f"\nIndex: {df.index}")
print(f"Columns: {df.columns}")
```
```

Index: Index(['Alice', 'Bob', 'Charlie', 'Diana'], dtype='object')
Columns: Index(['Score', 'Grade'], dtype='object')
```

```python
# Setting new index
df_new_index = df.set_index('Grade')
print("\nDataFrame with Grade as index:")
print(df_new_index)
```
```

DataFrame with Grade as index:
       Score
Grade       
B         85
A         92
C         78
A         96
```

```python
# Resetting index
df_reset = df_new_index.reset_index()
print("\nDataFrame with reset index:")
print(df_reset)
```
```

DataFrame with reset index:
  Grade  Score
0     B     85
1     A     92
2     C     78
3     A     96
```

### Basic DataFrame Operations

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12]
})

# Column operations
print("Column A:")
print(df['A'])              # Select single column (returns Series)
```
```
Column A:
0    1
1    2
2    3
3    4
Name: A, dtype: int64
```

```python
print("\nColumns A and C:")
print(df[['A', 'C']])       # Select multiple columns (returns DataFrame)
```
```

Columns A and C:
   A   C
0  1   9
1  2  10
2  3  11
3  4  12
```

```python
# Row operations
print("\nFirst row:")
print(df.iloc[0])           # Select first row
```
```

First row:
A    1
B    5
C    9
Name: 0, dtype: int64
```

```python
print("\nFirst two rows:")
print(df.iloc[0:2])         # Select multiple rows
```
```

First two rows:
   A  B  C
0  1  5  9
1  2  6 10
```

```python
# Adding new columns
df['D'] = df['A'] + df['B']  # New column from calculation
df['E'] = 100                # New column with constant value
print("\nDataFrame with new columns:")
print(df)
```
```

DataFrame with new columns:
   A  B   C  D    E
0  1  5   9  6  100
1  2  6  10  8  100
2  3  7  11 10  100
3  4  8  12 12  100
```
```

### Data Types in DataFrames

```python
# Different data types
df = pd.DataFrame({
    'Integer': [1, 2, 3, 4],
    'Float': [1.1, 2.2, 3.3, 4.4],
    'String': ['a', 'b', 'c', 'd'],
    'Boolean': [True, False, True, False],
    'Date': pd.date_range('2023-01-01', periods=4)
})

print("Data types:")
print(df.dtypes)
```
```
Data types:
Integer             int64
Float             float64
String             object
Boolean              bool
Date       datetime64[ns]
dtype: object
```

```python
print("\nMemory usage:")
print(df.memory_usage(deep=True))
```
```

Memory usage:
Index      128
Integer     32
Float       32
String     120
Boolean      4
Date        32
dtype: int64
```

```python
# Converting data types
df['Integer_as_Float'] = df['Integer'].astype(float)
df['String_as_Category'] = df['String'].astype('category')

print("\nAfter type conversion:")
print(df.dtypes)
```
```

After type conversion:
Integer                      int64
Float                      float64
String                      object
Boolean                       bool
Date               datetime64[ns]
Integer_as_Float           float64
String_as_Category        category
dtype: object
```
```

### Comparison: Series vs DataFrame

| Aspect | Series | DataFrame |
|--------|---------|-----------|
| Dimensions | 1D | 2D |
| Structure | Single column | Multiple columns |
| Index | Single index | Row and column index |
| Creation | From list, dict, array | From dict of lists, list of dicts |
| Access | s[index] | df[column] or df.loc[row, col] |
| Use Case | Single variable | Multiple variables/features |

### Best Practices

1. **Consistent naming**: Use meaningful column names
2. **Data types**: Choose appropriate data types for memory efficiency
3. **Index design**: Use meaningful indices when appropriate
4. **Documentation**: Add metadata using attributes like `name`

### Common Mistakes to Avoid

1. **Chained indexing**: Avoid `df[col1][col2]`, use `df.loc[row, col]`
2. **Copy vs View**: Understand when operations create copies
3. **Index alignment**: Be aware of automatic index alignment in operations
4. **Memory usage**: Monitor memory with large datasets

---

**Next: Data Input/Output Operations**