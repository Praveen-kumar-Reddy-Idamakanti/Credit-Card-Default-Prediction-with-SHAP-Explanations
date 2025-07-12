"""SHAP explanations and visualizations for the credit risk model."""

import numpy as np
import matplotlib.pyplot as plt
import shap

class SHAPExplainer:
    """Handles SHAP explanations and visualizations."""
    
    def __init__(self, model, feature_names):
        """Initialize with a trained model and feature names."""
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model.model)
        
    def explain(self, X, X_test_df, output_files):
        """Generate SHAP explanations and visualizations."""
        print("\nGenerating SHAP explanations...")
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # For binary classification, we'll use the SHAP values for class 1
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_plot = shap_values[1]
            expected_value = self.explainer.expected_value[1]
        else:
            shap_values_plot = shap_values
            expected_value = self.explainer.expected_value
        
        # Create summary plot
        self._create_summary_plot(shap_values_plot, X_test_df, output_files['shap_summary'])
        
        # Create bar plot of mean SHAP values
        self._create_bar_plot(shap_values_plot, X_test_df, output_files['shap_bar'])
        
        # Create visualizations for the first prediction
        self._create_prediction_visualizations(
            expected_value, 
            shap_values_plot, 
            X_test_df,
            output_files['shap_first_prediction'],
            output_files['shap_waterfall']
        )
    
    def _create_summary_plot(self, shap_values, X_test_df, output_file):
        """Create SHAP summary plot."""
        plt.figure(figsize=(12, 8))
        try:
            shap.summary_plot(
                shap_values,
                X_test_df,
                feature_names=self.feature_names,
                max_display=min(15, X_test_df.shape[1]),
                show=False,
                plot_size=(12, 8)
            )
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create summary plot: {str(e)}")
    
    def _create_bar_plot(self, shap_values, X_test_df, output_file):
        """Create bar plot of mean SHAP values."""
        plt.figure(figsize=(12, 6))
        try:
            shap.summary_plot(
                shap_values,
                X_test_df,
                feature_names=self.feature_names,
                plot_type='bar',
                max_display=min(15, X_test_df.shape[1]),
                show=False
            )
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create bar plot: {str(e)}")
    
    def _create_prediction_visualizations(self, expected_value, shap_values_plot, X_test_df, 
                                        first_pred_file, waterfall_file):
        """Create visualizations for the first prediction."""
        try:
            # Ensure expected_value is a scalar
            if hasattr(expected_value, '__iter__'):
                expected_value = float(expected_value[0]) if len(expected_value) > 0 else 0.0
            else:
                expected_value = float(expected_value)
            
            # Get the first prediction's SHAP values
            if len(shap_values_plot.shape) > 1:
                first_shap_values = shap_values_plot[0]
            else:
                first_shap_values = shap_values_plot
            
            # Create a bar plot of SHAP values for the first prediction
            self._create_first_prediction_plot(first_shap_values, X_test_df, first_pred_file)
            
            # Create a waterfall plot if possible
            self._create_waterfall_plot(expected_value, first_shap_values, X_test_df, waterfall_file)
            
        except Exception as e:
            print(f"Warning: Could not create prediction visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _create_first_prediction_plot(self, shap_values, X_test_df, output_file):
        """Create a bar plot of SHAP values for the first prediction."""
        try:
            # Convert to numpy array if it's not already
            shap_values = np.array(shap_values).flatten()
            
            # Get absolute SHAP values for sorting
            abs_shap = np.abs(shap_values)
            idx = np.argsort(-abs_shap)  # Sort in descending order
            
            # Get top features
            top_n = min(10, len(self.feature_names))
            idx = idx[:top_n]
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Get colors based on SHAP value sign
            colors = ['#1f77b4' if shap_values[i] > 0 else '#ff7f0e' for i in idx][::-1]
            
            # Create horizontal bar plot
            plt.barh(
                range(len(idx)),
                shap_values[idx][::-1],
                color=colors
            )
            
            # Add feature names and values
            plt.yticks(
                range(len(idx)),
                [f"{self.feature_names[i]} = {X_test_df.iloc[0, i]:.2f}" 
                 for i in idx][::-1]
            )
            
            plt.title('Top Features Affecting the First Prediction')
            plt.xlabel('SHAP Value (impact on model output)')
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create first prediction plot: {str(e)}")
    
    def _create_waterfall_plot(self, expected_value, shap_values, X_test_df, output_file):
        """Create a waterfall plot for the first prediction."""
        try:
            plt.figure(figsize=(12, 6))
            
            # Convert to numpy array if it's not already
            shap_values = np.array(shap_values).flatten()
            
            # Create a simple bar plot as an alternative to waterfall
            # Sort features by absolute SHAP value
            idx = np.argsort(-np.abs(shap_values))  # Sort in descending order of absolute value
            top_n = min(10, len(self.feature_names))
            idx = idx[:top_n]
            
            # Create bar plot
            colors = ['#1f77b4' if shap_values[i] > 0 else '#ff7f0e' for i in idx]
            plt.barh(
                range(len(idx)),
                shap_values[idx][::-1],
                color=colors[::-1]
            )
            
            # Add feature names and values
            plt.yticks(
                range(len(idx)),
                [f"{self.feature_names[i]}" for i in idx][::-1]
            )
            
            plt.title('Top SHAP Values for First Prediction')
            plt.xlabel('SHAP value (impact on model output)')
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create waterfall plot: {str(e)}")
            import traceback
            traceback.print_exc()
