import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def check_data_integrity(df):
    print("\n Checking for missing values after processing:")
    print(df.isnull().sum().sort_values(ascending=False).head(20))
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            elif df[col].dtype == 'float64' or df[col].dtype == 'int64':
                df[col] =df[col].fillna(df[col].median())
                
        else:
            print(f"Column {col} has no missing values.")
    
    print("\n Checking for missing values after processing:")
    print(df.isnull().sum().sort_values(ascending=False).head(20))


    print("\nChecking for number of unique values in each column:")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} unique values")

    print("\nChecking the data types of each column:")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")

