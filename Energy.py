import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('C:/Users/2273581/Downloads/Energy_dataset.csv')

# Display the shape of the dataset before outlier removal
print("Shape of the dataset before outlier removal:", df.shape)

# Remove outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# List of columns to remove outliers from
columns_to_check = ['energy_consumption_kwh', 'peak_hours_usage', 'off_peak_usage', 'renewable_energy_pct', 'billing_amount', 'household_size', 'temperature_avg']

# Remove outliers for each column
for column in columns_to_check:
    df = remove_outliers(df, column)

# Display the shape of the dataset after outlier removal
print("Shape of the dataset after outlier removal:", df.shape)

# Use .info() method to display feature data types
print(df.info())

# Visualize the data
plt.figure(figsize=(10, 6))

# Histogram of all numerical features
df.hist(bins=30, figsize=(20, 15))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Boxplot to visualize outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title('Boxplot of Numerical Features')
plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(df)
plt.suptitle('Pairplot of Numerical Features')
plt.show()

# Heatmap to visualize correlation between features (excluding non-numerical columns)
plt.figure(figsize=(10, 6))
numerical_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Feature Correlations')
plt.show()

# Additional Visualizations
# Plot 1: Total Energy Consumption by Region
plt.figure(figsize=(10, 6))
df.groupby('region')['energy_consumption_kwh'].sum().plot(kind='bar')
plt.title('Total Energy Consumption by Region')
plt.xlabel('Region')
plt.ylabel('Total Energy Consumption (kWh)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Average Renewable Energy Percentage by Region
plt.figure(figsize=(10, 6))
df.groupby('region')['renewable_energy_pct'].mean().plot(kind='bar', color='green')
plt.title('Average Renewable Energy Percentage by Region')
plt.xlabel('Region')
plt.ylabel('Average Renewable Energy Percentage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 3: Average Billing Amount by Income Bracket
plt.figure(figsize=(10, 6))
df.groupby('income_bracket')['billing_amount'].mean().plot(kind='bar', color='orange')
plt.title('Average Billing Amount by Income Bracket')
plt.xlabel('Income Bracket')
plt.ylabel('Average Billing Amount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 4: Household Size Distribution
plt.figure(figsize=(10, 6))
df['household_size'].value_counts().sort_index().plot(kind='bar', color='purple')
plt.title('Household Size Distribution')
plt.xlabel('Household Size')
plt.ylabel('Number of Households')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Plot 5: Smart Meter Installation by Region
plt.figure(figsize=(10, 6))
df.groupby('region')['smart_meter_installed'].sum().plot(kind='bar', color='red')
plt.title('Smart Meter Installation by Region')
plt.xlabel('Region')
plt.ylabel('Number of Smart Meters Installed')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 6: Time of Day Pricing Adoption by Region
plt.figure(figsize=(10, 6))
df.groupby('region')['time_of_day_pricing'].sum().plot(kind='bar', color='brown')
plt.title('Time of Day Pricing Adoption by Region')
plt.xlabel('Region')
plt.ylabel('Number of Households with Time of Day Pricing')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
