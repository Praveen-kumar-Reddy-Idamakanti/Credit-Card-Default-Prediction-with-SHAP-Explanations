import pandas as pd
import numpy as np

def explore_dataset(file_path):
    """
    Explore the dataset and print useful information
    """
    print(f"\n{'='*50}")
    print(f"Exploring dataset: {file_path}")
    print(f"{'='*50}\n")
    
    # Read the Excel file
    try:
        # Try reading with openpyxl first
        df = pd.read_excel(file_path, engine='openpyxl')
    except:
        # Fall back to xlrd if openpyxl fails
        df = pd.read_excel(file_path, engine='xlrd')
    
    # Clean the column names (remove extra spaces and make lowercase)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Print basic information
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head().to_string())
    
    print("\n" + "="*50)
    print("COLUMN INFORMATION")
    print("="*50)
    
    # Get column information
    column_info = pd.DataFrame({
        'column_name': df.columns,
        'data_type': df.dtypes,
        'missing_values': df.isnull().sum(),
        'unique_values': df.nunique(),
        'sample_values': df.iloc[0].values
    })
    
    print("\nColumn Details:")
    print(column_info.to_string())
    
    # Check for potential target columns (binary or categorical with few unique values)
    print("\n" + "="*50)
    print("POTENTIAL TARGET COLUMNS")
    print("="*50)
    
    potential_targets = []
    for col in df.columns:
        unique_vals = df[col].nunique()
        if 1 < unique_vals <= 5:  # Columns with 2-5 unique values
            value_counts = df[col].value_counts().to_dict()
            potential_targets.append({
                'column': col,
                'unique_values': unique_vals,
                'value_distribution': value_counts
            })
    
    if potential_targets:
        print("\nPotential target columns (binary or categorical with few unique values):")
        for i, target in enumerate(potential_targets, 1):
            print(f"\n{i}. Column: {target['column']}")
            print(f"   Unique values: {target['unique_values']}")
            print(f"   Value distribution:")
            for val, count in target['value_distribution'].items():
                print(f"     - {val}: {count} ({(count/len(df))*100:.1f}%)")
    else:
        print("\nNo obvious target columns found. Looking at all columns...")
        for col in df.columns:
            print(f"\nColumn: {col}")
            print(f"Type: {df[col].dtype}")
            print(f"Unique values: {df[col].nunique()}")
            if df[col].nunique() < 10:
                print("Value counts:")
                print(df[col].value_counts())
    
    print("\n" + "="*50)
    print("DATASET PREVIEW")
    print("="*50)
    print(df.head().to_string())

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python explore_data.py <path_to_excel_file>")
        print("Example: python explore_data.py 'default of credit card clients.xls'")
        sys.exit(1)
    
    file_path = sys.argv[1]
    explore_dataset(file_path)
