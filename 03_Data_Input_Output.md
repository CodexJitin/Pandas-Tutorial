# 1.3 Data Input/Output Operations

## Reading Data

### Reading CSV Files

```python
import pandas as pd

# Basic CSV reading
df = pd.read_csv('data.csv')
print(df.head())
```
**Output:**
```
   Name  Age      City  Salary
0   Alice   25  New York   50000
1     Bob   30    London   60000
2 Charlie   35     Paris   70000
3   Diana   28     Tokyo   55000
```

```python
# CSV with custom parameters
df = pd.read_csv('data.csv', 
                 sep=',',           # Separator
                 header=0,          # Header row
                 index_col=0,       # Use first column as index
                 skiprows=1,        # Skip first row
                 nrows=100,         # Read only first 100 rows
                 usecols=['Name', 'Age'],  # Read specific columns
                 dtype={'Age': int},       # Specify data types
                 na_values=['N/A', 'NULL'] # Additional missing values
                )
print(df.info())
```
**Output:**
```
<class 'pandas.core.frame.DataFrame'>
Index: 3 entries, Alice to Diana
Data columns (total 1 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   Age     3 non-null      int64
dtypes: int64(1)
memory usage: 48.0+ bytes
```

### Reading Excel Files

```python
# Basic Excel reading
df_excel = pd.read_excel('data.xlsx')
print(df_excel.head())
```
**Output:**
```
   Name  Age      City  Salary
0   Alice   25  New York   50000
1     Bob   30    London   60000
2 Charlie   35     Paris   70000
```

```python
# Excel with multiple sheets
df_sheet1 = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df_sheet2 = pd.read_excel('data.xlsx', sheet_name=1)  # By index

# Read all sheets
all_sheets = pd.read_excel('data.xlsx', sheet_name=None)
print(f"Sheet names: {list(all_sheets.keys())}")
```
**Output:**
```
Sheet names: ['Sheet1', 'Sheet2', 'Summary']
```

### Reading JSON Files

```python
# Reading JSON
df_json = pd.read_json('data.json')
print(df_json)
```
**Output:**
```
      Name  Age      City  Salary
0    Alice   25  New York   50000
1      Bob   30    London   60000
2  Charlie   35     Paris   70000
```

```python
# JSON with different orientations
# For records format: [{"Name": "Alice", "Age": 25}, ...]
df_records = pd.read_json('data.json', orient='records')

# For index format: {"0": {"Name": "Alice", "Age": 25}, ...}
df_index = pd.read_json('data.json', orient='index')

print("Records format:")
print(df_records.head(2))
```
**Output:**
```
Records format:
    Name  Age      City
0  Alice   25  New York
1    Bob   30    London
```

### Reading from URLs

```python
# Read CSV from URL
url = 'https://raw.githubusercontent.com/pandas-dev/pandas/main/pandas/tests/io/data/csv/iris.csv'
df_url = pd.read_csv(url)
print(f"Shape: {df_url.shape}")
print(df_url.columns.tolist())
```
**Output:**
```
Shape: (150, 5)
['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Name']
```

### Reading Other Formats

```python
# Parquet files (efficient for large datasets)
# df_parquet = pd.read_parquet('data.parquet')

# HDF5 files
# df_hdf = pd.read_hdf('data.h5', key='df')

# SQL databases
# from sqlalchemy import create_engine
# engine = create_engine('sqlite:///database.db')
# df_sql = pd.read_sql('SELECT * FROM table_name', engine)

# Clipboard (copy data from Excel/web)
# df_clipboard = pd.read_clipboard()

print("Various file formats supported!")
```
**Output:**
```
Various file formats supported!
```

## Writing Data

### Writing CSV Files

```python
# Create sample DataFrame
data = {
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
    'Price': [999.99, 25.50, 75.00, 299.99],
    'Stock': [15, 100, 50, 25]
}
df = pd.DataFrame(data)

# Basic CSV writing
df.to_csv('output.csv')
print("CSV file saved!")

# CSV with custom parameters
df.to_csv('output_custom.csv',
          index=False,          # Don't write row indices
          sep=';',              # Use semicolon separator
          columns=['Product', 'Price'],  # Write specific columns
          header=['Item', 'Cost'],       # Custom column names
          float_format='%.2f'   # Format floating point numbers
         )
print("Custom CSV file saved!")
```
**Output:**
```
CSV file saved!
Custom CSV file saved!
```

```python
# Check what was written
df_check = pd.read_csv('output_custom.csv', sep=';')
print(df_check)
```
**Output:**
```
      Item    Cost
0   Laptop  999.99
1    Mouse   25.50
2 Keyboard   75.00
3  Monitor  299.99
```

### Writing Excel Files

```python
# Basic Excel writing
df.to_excel('output.xlsx', index=False)
print("Excel file saved!")

# Excel with multiple sheets
with pd.ExcelWriter('multiple_sheets.xlsx') as writer:
    df.to_excel(writer, sheet_name='Products', index=False)
    df[df['Price'] > 50].to_excel(writer, sheet_name='Expensive', index=False)
    
print("Multi-sheet Excel file saved!")
```
**Output:**
```
Excel file saved!
Multi-sheet Excel file saved!
```

### Writing JSON Files

```python
# Write to JSON with different orientations
df.to_json('output_records.json', orient='records', indent=2)
df.to_json('output_index.json', orient='index', indent=2)

print("JSON files saved!")

# Check records format
with open('output_records.json', 'r') as f:
    print("Records format preview:")
    print(f.read()[:150] + "...")
```
**Output:**
```
JSON files saved!
Records format preview:
[
  {
    "Product":"Laptop",
    "Price":999.99,
    "Stock":15
  },
  {
    "Product":"Mouse",
    "Price":25.5,
    "Stock":100
  },
...
```

### Writing to Other Formats

```python
# Write to Parquet (efficient format)
# df.to_parquet('output.parquet')

# Write to HDF5
# df.to_hdf('output.h5', key='df', mode='w')

# Write to SQL database
# df.to_sql('products', engine, if_exists='replace', index=False)

# Copy to clipboard
df.head(2).to_clipboard(index=False)
print("Data copied to clipboard!")
```
**Output:**
```
Data copied to clipboard!
```

## File Handling Best Practices

### Error Handling

```python
import os

def safe_read_csv(filename):
    try:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            print(f"Successfully read {filename}")
            print(f"Shape: {df.shape}")
            return df
        else:
            print(f"File {filename} not found!")
            return None
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

# Test the function
df = safe_read_csv('nonexistent.csv')
```
**Output:**
```
File nonexistent.csv not found!
```

### Working with Large Files

```python
# Reading large files in chunks
def process_large_csv(filename, chunksize=1000):
    total_rows = 0
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        # Process each chunk
        total_rows += len(chunk)
        print(f"Processed chunk with {len(chunk)} rows")
    
    print(f"Total rows processed: {total_rows}")
    return total_rows

# Example usage (would work with large file)
# process_large_csv('large_file.csv')
print("Large file processing function defined!")
```
**Output:**
```
Large file processing function defined!
```

### File Format Comparison

```python
import time
import numpy as np

# Create a larger sample dataset
large_data = {
    'A': np.random.randn(10000),
    'B': np.random.randn(10000),
    'C': ['Category_' + str(i % 5) for i in range(10000)],
    'D': pd.date_range('2023-01-01', periods=10000, freq='1min')
}
large_df = pd.DataFrame(large_data)

print(f"Dataset shape: {large_df.shape}")
print(f"Memory usage: {large_df.memory_usage(deep=True).sum() / 1024:.1f} KB")

# Time different write operations
formats = {
    'CSV': lambda: large_df.to_csv('test.csv'),
    'JSON': lambda: large_df.to_json('test.json'),
    'Excel': lambda: large_df.to_excel('test.xlsx')
}

for format_name, write_func in formats.items():
    start_time = time.time()
    try:
        write_func()
        end_time = time.time()
        print(f"{format_name} write time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"{format_name} write failed: {e}")
```
**Output:**
```
Dataset shape: (10000, 4)
Memory usage: 781.3 KB
CSV write time: 0.15 seconds
JSON write time: 0.23 seconds
Excel write time: 1.45 seconds
```

### Encoding Issues

```python
# Handle different encodings
encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

def read_with_encoding(filename):
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(filename, encoding=encoding)
            print(f"Successfully read with {encoding} encoding")
            return df
        except UnicodeDecodeError:
            print(f"Failed with {encoding} encoding")
            continue
    
    print("All encodings failed!")
    return None

# Example usage
print("Encoding handler function defined!")
```
**Output:**
```
Encoding handler function defined!
```

## Summary of I/O Operations

| Format | Read Function | Write Method | Best For |
|--------|---------------|--------------|----------|
| CSV | `pd.read_csv()` | `.to_csv()` | Simple data, human-readable |
| Excel | `pd.read_excel()` | `.to_excel()` | Business reports, multiple sheets |
| JSON | `pd.read_json()` | `.to_json()` | Web APIs, nested data |
| Parquet | `pd.read_parquet()` | `.to_parquet()` | Large datasets, fast I/O |
| SQL | `pd.read_sql()` | `.to_sql()` | Database integration |

### Key Parameters to Remember

**For Reading:**
- `sep/delimiter`: Field separator
- `header`: Row to use as column names
- `index_col`: Column to use as row index
- `usecols`: Columns to read
- `dtype`: Data type specifications
- `na_values`: Additional strings to recognize as NaN

**For Writing:**
- `index`: Whether to write row names
- `columns`: Columns to write
- `sep`: Field separator
- `header`: Whether to write column names
- `mode`: Write mode ('w' for overwrite, 'a' for append)

---

**Next: Basic Data Exploration**