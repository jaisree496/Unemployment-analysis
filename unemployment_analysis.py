import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("unemployment.csv")

print("Dataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# Rename columns (because dataset has long column names)
df.columns = df.columns.str.strip()

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Drop missing values if any
df.dropna(inplace=True)

# Basic statistics
print("\nStatistical Summary:")
print(df.describe())

# -------------------------------------------
# Visualization 1: Unemployment rate over time
# -------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(df["Date"], df["Estimated Unemployment Rate (%)"], marker="o")
plt.title("Unemployment Rate Over Time in India")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.show()

# -------------------------------------------
# Visualization 2: Covid Impact Analysis
# Covid period (approx) 2020 March onwards
# -------------------------------------------
covid_df = df[df["Date"] >= "2020-03-01"]
before_covid_df = df[df["Date"] < "2020-03-01"]

plt.figure(figsize=(12,6))
plt.plot(before_covid_df["Date"], before_covid_df["Estimated Unemployment Rate (%)"], label="Before Covid")
plt.plot(covid_df["Date"], covid_df["Estimated Unemployment Rate (%)"], label="During Covid", color="red")
plt.title("Unemployment Rate Before vs During Covid")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------
# Visualization 3: State-wise unemployment rate
# -------------------------------------------
state_avg = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean().sort_values(ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x=state_avg.values, y=state_avg.index)
plt.title("Average Unemployment Rate by Region (State-wise)")
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("Region")
plt.show()

# -------------------------------------------
# Visualization 4: Heatmap (Correlation)
# -------------------------------------------
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------------------
# Insights
# -------------------------------------------
print("\nTop 5 Regions with Highest Unemployment Rate:")
print(state_avg.head())

print("\nTop 5 Regions with Lowest Unemployment Rate:")
print(state_avg.tail())

# -------------------------------------------
# Monthly Trend Analysis
# -------------------------------------------
df["Month"] = df["Date"].dt.month
monthly_avg = df.groupby("Month")["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(10,5))
sns.lineplot(x=monthly_avg.index, y=monthly_avg.values, marker="o")
plt.title("Monthly Average Unemployment Rate Trend")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.show()

print("\nMonthly Average Unemployment Rate:")
print(monthly_avg)
