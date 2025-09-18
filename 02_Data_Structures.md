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

# From a list with custom index
s2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s2)

# From a dictionary
data_dict = {'Apple': 100, 'Banana': 150, 'Orange': 80}
s3 = pd.Series(data_dict)
print(s3)

# From NumPy array
arr = np.array([1, 2, 3, 4])
s4 = pd.Series(arr, index=['w', 'x', 'y', 'z'])
print(s4)

# From scalar value
s5 = pd.Series(5, index=['a', 'b', 'c', 'd'])
print(s5)
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

# Setting name
s.name = "Sample Series"
s.index.name = "Letters"
print(s)
```

### Indexing and Accessing Data

```python
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])

# Label-based indexing
print(s['a'])           # Single element
print(s[['a', 'c']])    # Multiple elements

# Position-based indexing
print(s[0])             # First element
print(s[1:3])           # Slice

# Boolean indexing
print(s[s > 25])        # Elements greater than 25

# Using .loc and .iloc
print(s.loc['b'])       # Label-based
print(s.iloc[1])        # Position-based
```

### Basic Operations on Series

```python
s = pd.Series([1, 2, 3, 4, 5])

# Arithmetic operations
print(s + 10)           # Add scalar
print(s * 2)            # Multiply by scalar
print(s ** 2)           # Power

# Statistical operations
print(s.sum())          # Sum
print(s.mean())         # Mean
print(s.std())          # Standard deviation
print(s.describe())     # Descriptive statistics
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

# From list of dictionaries
data_list = [
    {'Name': 'Alice', 'Age': 25, 'City': 'New York'},
    {'Name': 'Bob', 'Age': 30, 'City': 'London'},
    {'Name': 'Charlie', 'Age': 35, 'City': 'Paris'}
]
df2 = pd.DataFrame(data_list)
print(df2)

# From NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df3 = pd.DataFrame(arr, columns=['A', 'B', 'C'], index=['X', 'Y', 'Z'])
print(df3)

# From Series
s1 = pd.Series([1, 2, 3], name='Column1')
s2 = pd.Series([4, 5, 6], name='Column2')
df4 = pd.DataFrame([s1, s2]).T  # Transpose to get correct orientation
print(df4)
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

# More detailed information
print(df.info())                    # Comprehensive info
print(df.describe())                # Statistical summary
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

# Accessing index and columns
print(f"\nIndex: {df.index}")
print(f"Columns: {df.columns}")

# Setting new index
df_new_index = df.set_index('Grade')
print("\nDataFrame with Grade as index:")
print(df_new_index)

# Resetting index
df_reset = df_new_index.reset_index()
print("\nDataFrame with reset index:")
print(df_reset)
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

print("\nColumns A and C:")
print(df[['A', 'C']])       # Select multiple columns (returns DataFrame)

# Row operations
print("\nFirst row:")
print(df.iloc[0])           # Select first row

print("\nFirst two rows:")
print(df.iloc[0:2])         # Select multiple rows

# Adding new columns
df['D'] = df['A'] + df['B']  # New column from calculation
df['E'] = 100                # New column with constant value
print("\nDataFrame with new columns:")
print(df)
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

print("\nMemory usage:")
print(df.memory_usage(deep=True))

# Converting data types
df['Integer_as_Float'] = df['Integer'].astype(float)
df['String_as_Category'] = df['String'].astype('category')

print("\nAfter type conversion:")
print(df.dtypes)
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