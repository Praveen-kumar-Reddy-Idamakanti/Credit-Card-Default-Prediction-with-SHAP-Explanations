# Explainable Credit Risk Prediction using SHAP

This project demonstrates how to build and interpret a credit risk prediction model using SHAP (SHapley Additive exPlanations) values. The goal is to create a model that not only predicts credit risk but also provides explanations for its predictions.

## Features

- Data preprocessing and feature engineering
- Handling class imbalance using SMOTE
- Random Forest Classifier for prediction
- Model evaluation with various metrics
- SHAP-based model interpretation
- Visualization of feature importance and individual predictions

## Requirements

- Python 3.7+
- Required packages are listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your credit risk dataset in CSV format
2. Update the `load_and_prepare_data` method in `credit_risk_prediction.py` to load your data
3. Run the script:
   ```
   python credit_risk_prediction.py
   ```
4. Check the generated plots for model insights:
   - `confusion_matrix.png`: Model performance visualization
   - `shap_summary.png`: Feature importance plot
   - `shap_force_plot.png`: Explanation for a single prediction

## Project Structure

- `credit_risk_prediction.py`: Main script containing the model and explanation logic
- `requirements.txt`: List of required Python packages
- `README.md`: This file

## Next Steps

- Add more sophisticated feature engineering
- Experiment with different models
- Create a web interface for interactive explanations
- Add more detailed documentation and examples
