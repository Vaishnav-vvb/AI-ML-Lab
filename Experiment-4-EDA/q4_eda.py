from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
DATA_PATH = Path(__file__).resolve().parent / "cleaned_all_phones.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATA PREPROCESSING AND EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 2. Understand the dataset
print("\n1. ORIGINAL DATASET")
print("\nFirst five records:")
print(df.head())

print("\nDataset dimensions:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe(include="all").T)

# 3. Identify missing values and duplicates
print("\n2. DATA QUALITY CHECK")
print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# 4. Remove duplicate records
original_rows = len(df)
df = df.drop_duplicates().copy()
removed_duplicates = original_rows - len(df)

# 5. Handle missing values
numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

for col in numeric_cols:
    if df[col].isnull().any():
        median_value = df[col].median()
        if pd.notna(median_value):
            df[col] = df[col].fillna(median_value)
        else:
            df[col] = df[col].fillna(0)

for col in categorical_cols:
    if df[col].isnull().any():
        mode_values = df[col].mode()
        if not mode_values.empty:
            df[col] = df[col].fillna(mode_values.iloc[0])
        else:
            df[col] = df[col].fillna("Unknown")

print("\n3. CLEANED DATASET")
print("\nShape after removing duplicates:", df.shape)
print("Duplicate rows removed:", removed_duplicates)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 6. Standardize numerical features
numeric_cols = df.select_dtypes(include="number").columns.tolist()
scaled_df = df.copy()

if numeric_cols:
    scaler = StandardScaler()
    scaled_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print("\n4. STANDARDIZED NUMERICAL FEATURES")
    print(scaled_df[numeric_cols].head().round(2))
else:
    print("\nNo numerical columns available for standardization.")

# 7. Exploratory Data Analysis and visualization
sns.set_theme(style="whitegrid")

# A. Histograms for up to four numerical columns
if numeric_cols:
    columns_to_plot = numeric_cols[:4]

    df[columns_to_plot].hist(
        figsize=(12, 8),
        bins=20,
        edgecolor="black"
    )
    plt.suptitle("Distribution of Numerical Features")
    plt.tight_layout()
    plt.show()

# B. Correlation heatmap
if len(numeric_cols) >= 2:
    correlation = df[numeric_cols].corr()

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        correlation,
        annot=len(numeric_cols) <= 10,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# C. Scatter plot of the first two numerical columns
if len(numeric_cols) >= 2:
    x_col, y_col = numeric_cols[:2]

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.title(f"{x_col} vs {y_col}")
    plt.tight_layout()
    plt.show()

# D. Distribution of the first categorical column
if categorical_cols:
    category_col = categorical_cols[0]
    top_categories = df[category_col].value_counts().head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=top_categories.values,
        y=top_categories.index
    )
    plt.title(f"Top Categories in {category_col}")
    plt.xlabel("Count")
    plt.ylabel(category_col)
    plt.tight_layout()
    plt.show()

# 8. Interpretation
print("\n5. INTERPRETATION")

print("\nDataset contains", len(df), "records and", len(df.columns), "columns.")
print("Numerical columns:", len(numeric_cols))
print("Categorical columns:", len(categorical_cols))

if numeric_cols:
    print("\nNumerical feature summary:")
    print(df[numeric_cols].describe().round(2))

if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr()
    pairs = corr.where(
        ~pd.DataFrame(
            [[i == j for j in range(len(corr.columns))]
             for i in range(len(corr.columns))],
            index=corr.index,
            columns=corr.columns
        )
    ).stack()

    if not pairs.empty:
        strongest_pair = pairs.abs().idxmax()
        value = corr.loc[strongest_pair[0], strongest_pair[1]]
        print(
            f"\nStrongest numerical correlation: "
            f"{strongest_pair[0]} and {strongest_pair[1]} = {value:.3f}"
        )
        print("Correlation indicates association, not causation.")

if categorical_cols:
    col = categorical_cols[0]
    print(f"\nMost frequent {col}: {df[col].mode().iloc[0]}")

# 9. Conclusion
print("\n6. CONCLUSION")
print("Data preprocessing and EDA completed successfully.")
print("Duplicate records were removed and missing values were handled.")
print("Numerical features were standardized for analysis.")
print("Visualizations were generated to identify data distributions")
print("and relationships between features.")
print("\nProgram executed successfully.")
