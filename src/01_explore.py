"""
Step 1: Explore the clinical trial dataset.
PURPOSE: Understand the data shape, columns, missing values, and messiness
before cleaning anything. Like opening a messy box to see what's inside.
"""

import pandas as pd

def explore(filepath):
    df = pd.read_csv(filepath)

    # Shape: how many rows and columns?
    print("=== SHAPE ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Column names
    print("\n=== COLUMNS ===")
    for col in df.columns:
        print(f"  - {col}")

    # First 3 rows
    print("\n=== FIRST 3 ROWS ===")
    print(df.head(3).to_string())

    # Missing values per column
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    # Unique value counts per column
    print("\n=== UNIQUE VALUES ===")
    for col in df.columns:
        print(f"  {col}: {df[col].nunique()} unique")

    # Multi-value rows check
    pipe = df['condition'].str.contains(r'\|').sum()
    plus = df['intervention_name'].str.contains(r'\+').sum()
    comma = df['country'].dropna().str.contains(',').sum()
    country_total = df['country'].notna().sum()

    print("\n=== MULTI-VALUE ROWS ===")
    print(f"  Country with comma:        {comma} / {country_total}")
    print(f"  Condition with pipe (|):   {pipe} / {len(df)}")
    print(f"  Intervention with (+):     {plus} / {len(df)}")

    # Sample messy values from target fields
    targets = ['intervention_name', 'condition', 'sponsor', 'country']
    print("\n=== SAMPLE VALUES (10 per field) ===")
    for col in targets:
        print(f"\n--- {col} ---")
        print(df[col].dropna().sample(10, random_state=42).tolist())

    return df


if __name__ == "__main__":
    explore("data/raw/trials_sample.csv")