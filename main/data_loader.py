"""Data loading and preprocessing module for credit risk prediction."""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

class DataLoader:
    """Handles loading and preprocessing of credit risk data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names_ = None
        
    def load_data(self, file_path, target_column, test_size=0.2, random_state=42):
        """Load data from file and split into train/test sets."""
        print(f"Loading data from {file_path}...")
        
        # Load the data
        if file_path.lower().endswith('.csv'):
            data = pd.read_csv(file_path)
        else:  # For Excel files
            try:
                data = pd.read_excel(file_path, engine='openpyxl', header=1, index_col=0)
            except:
                data = pd.read_excel(file_path, engine='xlrd', header=1, index_col=0)
        
        print(f"\nDataset shape: {data.shape}")
        print("\nFirst few rows of the dataset:")
        print(data.head())
        
        # Convert column names to lowercase
        data.columns = data.columns.str.lower()
        target_column = target_column.lower()
        
        # Check if target column exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset. "
                           f"Available columns: {data.columns.tolist()}")
        
        # Preprocess data
        X, y, feature_names = self._preprocess_data(data, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        print(f"\nClass distribution in training set: {dict(pd.Series(y_train).value_counts())}")
        
        # Handle class imbalance
        X_train_resampled, y_train_resampled = self._handle_class_imbalance(X_train, y_train, random_state)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_resampled)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train_resampled, y_test, feature_names
    
    def _preprocess_data(self, data, target_column):
        """Preprocess the data."""
        from config import DATA_PARAMS
        
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Converts non-numeric target labels to integers (e.g., "Yes"/"No" → 1/0)
        if df[target_column].dtype == 'object':
            df[target_column] = df[target_column].astype('category').cat.codes
        
        # Handle categorical variables
        categorical_cols = []
        for col in DATA_PARAMS['categorical_columns']:
            if col in df.columns:
                df[col] = df[col].astype('category')
                categorical_cols.append(col)
        
        # Get all feature names before one-hot encoding
        feature_names = [col for col in df.columns if col != target_column]
        
        # Convert categorical variables to dummy variables
        if categorical_cols:
            X = pd.get_dummies(df.drop(columns=[target_column]), 
                             columns=categorical_cols, 
                             drop_first=True)
        else:
            X = df.drop(columns=[target_column])
            
        y = df[target_column]
        
        # Store the column order for later use
        self.feature_names_ = X.columns.tolist()
        
        #X: processed feature matrix (with dummies if applicable)
        #y: binary target variable
        #feature_names: list of feature names (with dummies if applicable)
        
        return X, y, feature_names
    
    def _handle_class_imbalance(self, X, y, random_state):
        """Handle class imbalance using SMOTE if needed."""
        class_distribution = y.value_counts(normalize=True).mul(100).round(1)
        print("\nClass distribution in training set:")
        print(class_distribution.astype(str) + '%')
        
        # Only apply SMOTE if there's significant class imbalance
        if class_distribution.min() / 100 < 0.3:  # If minority class < 30%
            print("\nHandling class imbalance using SMOTE...")
            smote = SMOTE(random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print(f"After SMOTE - Class distribution: {dict(pd.Series(y_resampled).value_counts(normalize=True).mul(100).round(1).astype(str) + '%')}")
            return X_resampled, y_resampled
        
        print("\nClass distribution is balanced. Skipping SMOTE.")
        return X, y
