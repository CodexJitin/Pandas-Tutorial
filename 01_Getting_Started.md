# 1.1 Getting Started with Pandas

## Installation and Setup

### Installing Pandas

Pandas can be installed using various package managers:

```bash
# Using pip
pip install pandas

# Using conda
conda install pandas

# Install with additional dependencies
pip install pandas[all]
```

### Importing Pandas

```python
import pandas as pd
import numpy as np  # Often used together with pandas

# Check pandas version
print(pd.__version__)
```

### Setting Up Environment

```python
# Common imports for data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', 100)     # Show up to 100 rows
pd.set_option('display.width', None)       # Don't wrap output
pd.set_option('display.max_colwidth', 50)  # Max column width

# Reset options to default
pd.reset_option('all')
```

### Key Libraries and Dependencies

1. **NumPy**: Foundation for numerical operations
2. **Matplotlib**: Basic plotting capabilities
3. **Seaborn**: Statistical visualization
4. **Openpyxl**: Excel file support
5. **SQLAlchemy**: Database connectivity

### Development Environment Setup

```python
# Jupyter Notebook magic commands
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# For better display in Jupyter
from IPython.display import display, HTML
pd.set_option('display.notebook_repr_html', True)
```

### Common Initial Setup Script

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Configure pandas
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
pd.set_option('display.precision', 2)

# Configure matplotlib
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)

# Suppress warnings
warnings.filterwarnings('ignore')

print("Pandas setup complete!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
```

### Troubleshooting Common Installation Issues

1. **ImportError**: Ensure all dependencies are installed
2. **Version conflicts**: Use virtual environments
3. **Permission errors**: Use `--user` flag with pip
4. **Memory issues**: Consider lightweight alternatives for large datasets

### Best Practices for Setup

1. Use virtual environments for projects
2. Pin specific versions in requirements.txt
3. Import pandas as 'pd' (standard convention)
4. Set display options based on your needs
5. Keep dependencies minimal for production code

---

**Next: Data Structures - Series and DataFrames**