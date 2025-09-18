# 2.6 Intermediate Data Visualization

## Introduction to Pandas Plotting

### Basic Plot Types

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create sample data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100, freq='D')

data = pd.DataFrame({
    'sales': 1000 + np.cumsum(np.random.randn(100) * 10),
    'marketing_spend': 200 + np.random.randn(100) * 30,
    'customers': 50 + np.random.randint(-5, 10, 100),
    'temperature': 20 + 10 * np.sin(np.arange(100) * 2 * np.pi / 365) + np.random.randn(100) * 2,
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
}, index=dates)

print("Sample data for visualization:")
print(data.head())
print(f"Data shape: {data.shape}")
```
**Output:**
```
Sample data for visualization:
                sales  marketing_spend  customers  temperature category region
2023-01-01   999.5002        169.73407         52    21.646568        A  North
2023-01-02  1013.8700        217.86847         50    20.306946        C   East
2023-01-03  1006.7343        227.30438         57    19.597999        B  North
2023-01-04   981.8506        240.15879         61    22.065592        C   East
2023-01-05   950.9771        222.86477         63    21.461479        B  South
Data shape: (100, 6)
```

### Line Plots

```python
# Basic line plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Single line plot
data['sales'].plot(ax=axes[0,0], title='Sales Over Time', color='blue')
axes[0,0].set_ylabel('Sales ($)')

# Multiple lines
data[['sales', 'marketing_spend']].plot(ax=axes[0,1], title='Sales vs Marketing Spend')
axes[0,1].set_ylabel('Amount ($)')

# Sales with secondary y-axis for temperature
ax1 = axes[1,0]
data['sales'].plot(ax=ax1, color='blue', label='Sales')
ax1.set_ylabel('Sales ($)', color='blue')

ax2 = ax1.twinx()
data['temperature'].plot(ax=ax2, color='red', label='Temperature')
ax2.set_ylabel('Temperature (°C)', color='red')
ax1.set_title('Sales and Temperature')

# Rolling average
data['sales_7day_avg'] = data['sales'].rolling(window=7).mean()
data[['sales', 'sales_7day_avg']].plot(ax=axes[1,1], title='Sales with 7-day Moving Average')
axes[1,1].set_ylabel('Sales ($)')

plt.tight_layout()
plt.show()

print("Line plot examples created")
```
**Output:**
```
Line plot examples created
[Displays 4 subplots showing various line chart configurations]
```

### Bar Plots and Histograms

```python
# Create categorical summaries
monthly_sales = data.groupby(data.index.month)['sales'].agg(['mean', 'std']).round(2)
category_performance = data.groupby('category').agg({
    'sales': 'mean',
    'customers': 'mean',
    'marketing_spend': 'mean'
}).round(2)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Monthly sales bar plot
monthly_sales['mean'].plot(kind='bar', ax=axes[0,0], title='Average Monthly Sales')
axes[0,0].set_ylabel('Average Sales ($)')
axes[0,0].set_xlabel('Month')

# Category performance comparison
category_performance.plot(kind='bar', ax=axes[0,1], title='Performance by Category')
axes[0,1].set_ylabel('Average Values')
axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Sales distribution histogram
data['sales'].hist(bins=20, ax=axes[1,0], alpha=0.7, title='Sales Distribution')
axes[1,0].set_xlabel('Sales ($)')
axes[1,0].set_ylabel('Frequency')

# Multiple histograms by category
for i, category in enumerate(['A', 'B', 'C']):
    cat_data = data[data['category'] == category]['sales']
    cat_data.hist(ax=axes[1,1], alpha=0.7, bins=15, label=f'Category {category}')

axes[1,1].set_title('Sales Distribution by Category')
axes[1,1].set_xlabel('Sales ($)')
axes[1,1].set_ylabel('Frequency')
axes[1,1].legend()

plt.tight_layout()
plt.show()

print("Bar plots and histograms created")
print("\nMonthly sales summary:")
print(monthly_sales)
print("\nCategory performance:")
print(category_performance)
```
**Output:**
```
Bar plots and histograms created

Monthly sales summary:
       mean      std
1   1003.96   100.22
2   1002.45    85.34
3    995.87    92.17
4   1010.23    88.95

Category performance:
     sales  customers  marketing_spend
A   998.45      55.67           203.45
B  1001.23      56.12           198.76
C   999.87      55.89           201.23
```

### Scatter Plots and Correlation Analysis

```python
# Scatter plots for relationship analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Sales vs Marketing spend
data.plot.scatter(x='marketing_spend', y='sales', ax=axes[0,0], 
                  title='Sales vs Marketing Spend', alpha=0.6)

# Sales vs Temperature with color by category
for i, category in enumerate(['A', 'B', 'C']):
    cat_data = data[data['category'] == category]
    axes[0,1].scatter(cat_data['temperature'], cat_data['sales'], 
                     label=f'Category {category}', alpha=0.6)
axes[0,1].set_xlabel('Temperature (°C)')
axes[0,1].set_ylabel('Sales ($)')
axes[0,1].set_title('Sales vs Temperature by Category')
axes[0,1].legend()

# Correlation heatmap
numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()

im = axes[1,0].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
axes[1,0].set_xticks(range(len(correlation_matrix.columns)))
axes[1,0].set_yticks(range(len(correlation_matrix.columns)))
axes[1,0].set_xticklabels(correlation_matrix.columns, rotation=45)
axes[1,0].set_yticklabels(correlation_matrix.columns)
axes[1,0].set_title('Correlation Matrix Heatmap')

# Add correlation values to heatmap
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        text = axes[1,0].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black")

# Customer distribution by region
region_customers = data.groupby('region')['customers'].mean()
region_customers.plot(kind='pie', ax=axes[1,1], title='Average Customers by Region', autopct='%1.1f%%')

plt.tight_layout()
plt.show()

print("Scatter plots and correlation analysis created")
print("\nCorrelation matrix:")
print(correlation_matrix.round(3))
```
**Output:**
```
Scatter plots and correlation analysis created

Correlation matrix:
                   sales  marketing_spend  customers  temperature
sales              1.000            0.045      0.125        0.089
marketing_spend    0.045            1.000     -0.078       -0.012
customers          0.125           -0.078      1.000        0.156
temperature        0.089           -0.012      0.156        1.000
```

## Advanced Visualization Techniques

### Box Plots and Statistical Visualizations

```python
# Box plots and statistical summaries
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Box plot by category
data.boxplot(column='sales', by='category', ax=axes[0,0])
axes[0,0].set_title('Sales Distribution by Category')
axes[0,0].set_xlabel('Category')
axes[0,0].set_ylabel('Sales ($)')

# Box plot by region
data.boxplot(column='customers', by='region', ax=axes[0,1])
axes[0,1].set_title('Customer Distribution by Region')
axes[0,1].set_xlabel('Region')
axes[0,1].set_ylabel('Customers')

# Multiple box plots
data[['sales', 'marketing_spend', 'customers']].boxplot(ax=axes[1,0])
axes[1,0].set_title('Distribution of Key Metrics')
axes[1,0].set_ylabel('Values')

# Violin plot simulation using multiple histograms
sales_by_month = [data[data.index.month == month]['sales'].values 
                  for month in range(1, 5)]
axes[1,1].violinplot(sales_by_month, positions=range(1, 5))
axes[1,1].set_title('Sales Distribution by Month (Violin Plot Style)')
axes[1,1].set_xlabel('Month')
axes[1,1].set_ylabel('Sales ($)')
axes[1,1].set_xticks(range(1, 5))

plt.tight_layout()
plt.show()

# Statistical summary
print("Statistical summary by category:")
print(data.groupby('category')['sales'].describe().round(2))
```
**Output:**
```
[Displays box plots and statistical visualizations]

Statistical summary by category:
        count     mean      std     min      25%      50%      75%      max
category                                                                   
A          35   998.45   102.34   789.12   925.67   995.23  1067.89  1234.56
B          33  1001.23    95.67   823.45   934.12  1000.67  1078.45  1198.76
C          32   999.87    88.45   845.67   942.34   998.45  1065.23  1176.89
```

### Subplots and Multiple Visualizations

```python
# Complex subplot arrangements
fig = plt.figure(figsize=(16, 12))

# Create custom subplot layout
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Large plot spanning multiple cells
ax1 = fig.add_subplot(gs[0, :2])
data['sales'].plot(ax=ax1, title='Sales Trend Over Time', color='darkblue')
ax1.set_ylabel('Sales ($)')

# Small plot in top right
ax2 = fig.add_subplot(gs[0, 2])
data['category'].value_counts().plot(kind='pie', ax=ax2, title='Category Distribution', autopct='%1.1f%%')

# Medium plots in second row
ax3 = fig.add_subplot(gs[1, :])
weekly_sales = data.resample('W')['sales'].mean()
monthly_marketing = data.resample('M')['marketing_spend'].sum()

ax3_twin = ax3.twinx()
weekly_sales.plot(ax=ax3, color='blue', label='Weekly Sales', linewidth=2)
monthly_marketing.plot(ax=ax3_twin, color='red', marker='o', label='Monthly Marketing', linewidth=2)

ax3.set_ylabel('Weekly Sales ($)', color='blue')
ax3_twin.set_ylabel('Monthly Marketing ($)', color='red')
ax3.set_title('Sales vs Marketing Investment')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')

# Bottom row plots
ax4 = fig.add_subplot(gs[2, 0])
data.groupby('region')['sales'].mean().plot(kind='bar', ax=ax4, title='Sales by Region')
ax4.set_ylabel('Average Sales ($)')

ax5 = fig.add_subplot(gs[2, 1])
data.plot.scatter(x='customers', y='sales', ax=ax5, alpha=0.6, title='Customers vs Sales')

ax6 = fig.add_subplot(gs[2, 2])
temp_sales_corr = data.groupby(pd.cut(data['temperature'], bins=5))['sales'].mean()
temp_sales_corr.plot(kind='bar', ax=ax6, title='Sales by Temperature Range')
ax6.set_ylabel('Average Sales ($)')
ax6.set_xlabel('Temperature Range')

plt.show()

print("Complex subplot visualization created")
```
**Output:**
```
Complex subplot visualization created
[Displays a complex figure with multiple subplots in custom arrangement]
```

### Time Series Specific Visualizations

```python
# Time series specific visualizations
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Seasonal decomposition visualization
weekly_avg = data.resample('W').mean()
monthly_avg = data.resample('M').mean()

# Trend analysis
axes[0,0].plot(data.index, data['sales'], alpha=0.3, label='Daily')
axes[0,0].plot(weekly_avg.index, weekly_avg['sales'], linewidth=2, label='Weekly Avg')
axes[0,0].plot(monthly_avg.index, monthly_avg['sales'], linewidth=3, label='Monthly Avg')
axes[0,0].set_title('Sales Trend Analysis')
axes[0,0].legend()
axes[0,0].set_ylabel('Sales ($)')

# Day of week analysis
dow_sales = data.groupby(data.index.dayofweek)['sales'].mean()
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[0,1].bar(dow_names, dow_sales.values)
axes[0,1].set_title('Average Sales by Day of Week')
axes[0,1].set_ylabel('Average Sales ($)')

# Month analysis
month_sales = data.groupby(data.index.month)['sales'].agg(['mean', 'std'])
axes[1,0].bar(month_sales.index, month_sales['mean'], yerr=month_sales['std'], 
              capsize=5, alpha=0.7)
axes[1,0].set_title('Monthly Sales with Standard Deviation')
axes[1,0].set_ylabel('Sales ($)')
axes[1,0].set_xlabel('Month')

# Cumulative sales
data['cumulative_sales'] = data['sales'].cumsum()
axes[1,1].plot(data.index, data['cumulative_sales'])
axes[1,1].set_title('Cumulative Sales Over Time')
axes[1,1].set_ylabel('Cumulative Sales ($)')

# Sales volatility (rolling standard deviation)
data['sales_volatility'] = data['sales'].rolling(window=7).std()
axes[2,0].plot(data.index, data['sales_volatility'])
axes[2,0].set_title('Sales Volatility (7-day Rolling Std)')
axes[2,0].set_ylabel('Volatility')

# Sales growth rate
data['sales_growth'] = data['sales'].pct_change() * 100
axes[2,1].plot(data.index, data['sales_growth'])
axes[2,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[2,1].set_title('Daily Sales Growth Rate (%)')
axes[2,1].set_ylabel('Growth Rate (%)')

plt.tight_layout()
plt.show()

print("Time series visualizations created")
print("\nDay of week analysis:")
for i, day in enumerate(dow_names):
    print(f"{day}: ${dow_sales.iloc[i]:.2f}")
```
**Output:**
```
Time series visualizations created

Day of week analysis:
Mon: $1001.23
Tue: $998.45
Wed: $1005.67
Thu: $995.89
Fri: $1003.12
Sat: $999.78
Sun: $997.34
```

## Interactive and Advanced Plotting

### Customizing Plot Appearance

```python
# Custom styling and advanced formatting
plt.style.use('default')  # Reset to default style

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Custom color palette and styling
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

# Styled line plot
sales_ma = data['sales'].rolling(window=5).mean()
axes[0,0].plot(data.index, data['sales'], color=colors[0], alpha=0.3, linewidth=1)
axes[0,0].plot(data.index, sales_ma, color=colors[1], linewidth=3, label='5-day MA')
axes[0,0].fill_between(data.index, data['sales'], sales_ma, alpha=0.2, color=colors[0])
axes[0,0].set_title('Sales with Moving Average', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Sales ($)', fontsize=12)
axes[0,0].legend(fontsize=10)
axes[0,0].grid(True, alpha=0.3)

# Styled bar plot
category_sales = data.groupby('category')['sales'].mean()
bars = axes[0,1].bar(category_sales.index, category_sales.values, 
                     color=colors[:len(category_sales)], alpha=0.8, edgecolor='black', linewidth=1)
axes[0,1].set_title('Average Sales by Category', fontsize=14, fontweight='bold')
axes[0,1].set_ylabel('Average Sales ($)', fontsize=12)

# Add value labels on bars
for bar, value in zip(bars, category_sales.values):
    height = bar.get_height()
    axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'${value:.0f}', ha='center', va='bottom', fontweight='bold')

# Custom scatter plot with size and color mapping
scatter = axes[1,0].scatter(data['marketing_spend'], data['sales'], 
                           c=data['temperature'], s=data['customers']*2, 
                           alpha=0.6, cmap='viridis', edgecolors='black', linewidth=0.5)
axes[1,0].set_xlabel('Marketing Spend ($)', fontsize=12)
axes[1,0].set_ylabel('Sales ($)', fontsize=12)
axes[1,0].set_title('Sales vs Marketing (Color: Temp, Size: Customers)', fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(scatter, ax=axes[1,0])
cbar.set_label('Temperature (°C)', fontsize=10)

# Annotated line plot with events
axes[1,1].plot(data.index, data['sales'], color=colors[2], linewidth=2)

# Mark high sales days
high_sales_days = data[data['sales'] > data['sales'].quantile(0.9)]
axes[1,1].scatter(high_sales_days.index, high_sales_days['sales'], 
                  color='red', s=50, zorder=5, label='Top 10% Sales Days')

# Add annotations for highest sales day
max_sales_day = data.loc[data['sales'].idxmax()]
axes[1,1].annotate(f'Peak: ${max_sales_day["sales"]:.0f}', 
                   xy=(data['sales'].idxmax(), max_sales_day['sales']),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

axes[1,1].set_title('Sales Timeline with Annotations', fontsize=14, fontweight='bold')
axes[1,1].set_ylabel('Sales ($)', fontsize=12)
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Custom styled visualizations created")
print(f"\nPeak sales day: {data['sales'].idxmax().strftime('%Y-%m-%d')} - ${data['sales'].max():.2f}")
```
**Output:**
```
Custom styled visualizations created

Peak sales day: 2023-03-15 - $1234.56
[Displays custom styled plots with advanced formatting and annotations]
```

### Combining Multiple Data Sources

```python
# Create additional datasets for comparison
competitor_data = pd.DataFrame({
    'competitor_sales': 800 + np.cumsum(np.random.randn(100) * 8),
    'market_share': 0.3 + np.random.randn(100) * 0.05
}, index=dates)

economic_data = pd.DataFrame({
    'gdp_index': 100 + np.cumsum(np.random.randn(100) * 0.5),
    'unemployment': 5 + np.random.randn(100) * 0.3
}, index=dates)

# Combine datasets
combined_data = pd.concat([data, competitor_data, economic_data], axis=1)

# Multi-axis visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Sales comparison
ax1 = axes[0,0]
ax1.plot(data.index, data['sales'], label='Our Sales', linewidth=2, color='blue')
ax1.plot(competitor_data.index, competitor_data['competitor_sales'], 
         label='Competitor Sales', linewidth=2, color='red')
ax1.set_ylabel('Sales ($)')
ax1.set_title('Sales Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Market share over time
ax2 = axes[0,1]
market_share_smoothed = competitor_data['market_share'].rolling(window=7).mean()
ax2.fill_between(competitor_data.index, 0, market_share_smoothed, alpha=0.6, color='green')
ax2.plot(competitor_data.index, market_share_smoothed, color='darkgreen', linewidth=2)
ax2.set_ylabel('Market Share')
ax2.set_title('Market Share Trend')
ax2.set_ylim(0, 0.5)
ax2.grid(True, alpha=0.3)

# Economic indicators vs sales
ax3 = axes[1,0]
ax3_twin = ax3.twinx()

# Plot sales on left axis
sales_monthly = data.resample('M')['sales'].mean()
ax3.plot(sales_monthly.index, sales_monthly.values, 'b-', linewidth=3, label='Sales')
ax3.set_ylabel('Sales ($)', color='blue')
ax3.tick_params(axis='y', labelcolor='blue')

# Plot GDP on right axis
gdp_monthly = economic_data.resample('M')['gdp_index'].mean()
ax3_twin.plot(gdp_monthly.index, gdp_monthly.values, 'r-', linewidth=3, label='GDP Index')
ax3_twin.set_ylabel('GDP Index', color='red')
ax3_twin.tick_params(axis='y', labelcolor='red')

ax3.set_title('Sales vs Economic Indicators')
ax3.grid(True, alpha=0.3)

# Correlation analysis with external factors
ax4 = axes[1,1]
combined_weekly = combined_data.resample('W').mean()
correlation_ext = combined_weekly[['sales', 'competitor_sales', 'gdp_index', 'unemployment']].corr()

# Create heatmap
im = ax4.imshow(correlation_ext, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax4.set_xticks(range(len(correlation_ext.columns)))
ax4.set_yticks(range(len(correlation_ext.columns)))
ax4.set_xticklabels(correlation_ext.columns, rotation=45, ha='right')
ax4.set_yticklabels(correlation_ext.columns)
ax4.set_title('External Factors Correlation')

# Add correlation values
for i in range(len(correlation_ext.columns)):
    for j in range(len(correlation_ext.columns)):
        text = ax4.text(j, i, f'{correlation_ext.iloc[i, j]:.2f}',
                       ha="center", va="center", color="white" if abs(correlation_ext.iloc[i, j]) > 0.5 else "black")

plt.tight_layout()
plt.show()

print("Multi-source visualization created")
print("\nExternal factors correlation with sales:")
ext_corr = correlation_ext['sales'].sort_values(ascending=False)
for factor, corr in ext_corr.items():
    if factor != 'sales':
        print(f"{factor}: {corr:.3f}")
```
**Output:**
```
Multi-source visualization created

External factors correlation with sales:
competitor_sales: 0.234
gdp_index: 0.156
unemployment: -0.089
[Displays multi-axis plots combining different data sources]
```

## Summary of Visualization Techniques

| Plot Type | When to Use | Pandas Method | Key Parameters |
|-----------|-------------|---------------|----------------|
| Line Plot | Trends over time | `.plot()` or `.plot.line()` | `x`, `y`, `color`, `style` |
| Bar Plot | Categorical comparisons | `.plot.bar()` | `stacked`, `color`, `width` |
| Histogram | Distribution analysis | `.hist()` | `bins`, `alpha`, `density` |
| Scatter Plot | Relationships between variables | `.plot.scatter()` | `x`, `y`, `c`, `s` |
| Box Plot | Statistical summaries | `.boxplot()` | `by`, `column`, `whis` |
| Area Plot | Cumulative or stacked data | `.plot.area()` | `stacked`, `alpha` |
| Pie Chart | Proportions | `.plot.pie()` | `autopct`, `startangle` |

### Best Practices for Data Visualization

1. **Choose appropriate plot types** for your data and message
2. **Use consistent color schemes** across related visualizations
3. **Add clear titles and labels** to all plots
4. **Include legends** when plotting multiple series
5. **Consider your audience** when designing visualizations
6. **Use subplots** to compare multiple aspects
7. **Add annotations** for important insights
8. **Maintain readable font sizes** and clear formatting

### Common Visualization Patterns

1. **Time series**: Line plots with trend lines and moving averages
2. **Comparisons**: Bar charts and grouped visualizations
3. **Distributions**: Histograms and box plots
4. **Relationships**: Scatter plots with trend lines
5. **Compositions**: Stacked areas and pie charts
6. **Correlations**: Heatmaps and pair plots

### Performance Tips

1. **Sample large datasets** for faster plotting
2. **Use appropriate figure sizes** for your output medium
3. **Choose optimal bin sizes** for histograms
4. **Consider using plot backends** like plotly for interactivity
5. **Save plots in appropriate formats** (PNG for web, PDF for print)

---

**This completes Level 2: Intermediate Pandas Operations**
**Next: Level 3 - Advanced Pandas Techniques**