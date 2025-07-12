"""Configuration parameters for the credit risk prediction model."""

# Model parameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'class_weight': 'balanced'
}

# Data processing parameters
DATA_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'target_column': 'default payment next month',
    'categorical_columns': ['sex', 'education', 'marriage']
}

# Visualization parameters
VISUALIZATION_PARAMS = {
    'max_display': 15,
    'plot_style': 'default',
    'dpi': 300
}

import os

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# File paths
FILE_PATHS = {
    'output_dir': 'output',
    'confusion_matrix': os.path.join('output', 'confusion_matrix.png'),
    'shap_summary': os.path.join('output', 'shap_summary.png'),
    'shap_bar': os.path.join('output', 'shap_bar.png'),
    'shap_first_prediction': os.path.join('output', 'shap_first_prediction.png'),
    'shap_waterfall': os.path.join('output', 'shap_waterfall_plot.png')
}
