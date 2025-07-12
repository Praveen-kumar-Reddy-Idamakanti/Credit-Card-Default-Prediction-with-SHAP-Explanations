"""Main script for credit risk prediction with SHAP explanations."""

import argparse
import os
from data_loader import DataLoader
from model import CreditRiskModel
from explainer import SHAPExplainer
from config import DATA_PARAMS, VISUALIZATION_PARAMS, FILE_PATHS

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Credit Card Default Prediction with SHAP Explanations')
    parser.add_argument('--data', type=str, default='default of credit card clients.xls',
                      help='Path to the dataset file (default: default of credit card clients.xls)')
    parser.add_argument('--test_size', type=float, default=DATA_PARAMS['test_size'],
                      help=f'Proportion of data to use for testing (default: {DATA_PARAMS["test_size"]})')
    parser.add_argument('--random_state', type=int, default=DATA_PARAMS['random_state'],
                      help=f'Random seed for reproducibility (default: {DATA_PARAMS["random_state"]})')
    parser.add_argument('--max_display', type=int, default=VISUALIZATION_PARAMS['max_display'],
                      help=f'Maximum number of features to display in SHAP plots (default: {VISUALIZATION_PARAMS["max_display"]})')
    
    return parser.parse_args()

def main():
    """Main function to run the credit risk prediction pipeline."""
    args = parse_arguments()
    
    print("\n" + "="*50)
    print("Loading and preparing data...")
    print(f"Dataset: {args.data}")
    print(f"Target column: {DATA_PARAMS['target_column']}")
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")
    print("="*50 + "\n")
    
    try:
        # Load and preprocess data
        data_loader = DataLoader()
        X_train, X_test, y_train, y_test = data_loader.load_data(
            file_path=args.data,
            target_column=DATA_PARAMS['target_column'],
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Train model
        print("\nTraining model...")
        model = CreditRiskModel()
        model.train(X_train, y_train)
        
        # Evaluate model
        print("\nEvaluating model...")
        model.evaluate(X_test, y_test, output_file=FILE_PATHS['confusion_matrix'])
        
        # Generate explanations
        print("\nGenerating explanations...")
        explainer = SHAPExplainer(model, data_loader.feature_names_)
        explainer.explain(X_test, X_test, FILE_PATHS)
        
        print("\n" + "="*50)
        print("Model training and evaluation complete!")
        print("Check the following generated files:")
        print(f"- {FILE_PATHS['confusion_matrix']}: Model performance metrics")
        print(f"- {FILE_PATHS['shap_summary']}: SHAP summary plot (feature importance and impact)")
        print(f"- {FILE_PATHS['shap_bar']}: Mean absolute SHAP values (feature importance)")
        print(f"- {FILE_PATHS['shap_first_prediction']}: Top features for the first prediction")
        print(f"- {FILE_PATHS['shap_waterfall']}: Waterfall plot for the first prediction")
        
        print("\nKey Insights:")
        print("1. Check the SHAP summary plot to see how each feature affects the prediction")
        print("2. Look at the bar plot to identify the most important features")
        print("3. Review the confusion matrix for model performance metrics")
        print("\n" + "="*50 + "\n")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease check the following:")
        print("1. The file path is correct")
        print("2. The file is in a supported format (.csv, .xls, .xlsx)")
        print("3. The specified target column exists in the data")
        print("4. The data is properly formatted")

if __name__ == "__main__":
    main()
