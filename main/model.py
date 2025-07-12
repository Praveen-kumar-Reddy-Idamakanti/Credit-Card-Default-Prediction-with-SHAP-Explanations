"""Model definition for credit risk prediction."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class CreditRiskModel:
    """Random Forest model for credit risk prediction."""
    
    def __init__(self):
        """Initialize the model with default parameters."""
        from config import MODEL_PARAMS
        self.model = RandomForestClassifier(**MODEL_PARAMS)
        self.feature_importances_ = None
        
    def train(self, X_train, y_train):
        """Train the model on the given data."""
        print("Training model...")
        self.model.fit(X_train, y_train)
        self.feature_importances_ = self.model.feature_importances_
        return self
        
    def predict(self, X):
        """Make predictions on new data."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for X."""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, output_file='confusion_matrix.png'):
        """Evaluate the model and generate performance metrics."""
        print("\nModel Evaluation:")
        y_pred = self.predict(X_test)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Generate and save confusion matrix
        self._plot_confusion_matrix(y_test, y_pred, output_file)
        
        return classification_report(y_test, y_pred, output_dict=True)
    
    def _plot_confusion_matrix(self, y_true, y_pred, output_file):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
