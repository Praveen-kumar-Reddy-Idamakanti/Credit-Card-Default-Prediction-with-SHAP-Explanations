# Credit Card Default Prediction with SHAP Explanations

This project implements a machine learning solution for predicting credit card default risk while providing transparent and interpretable predictions using SHAP (SHapley Additive exPlanations) values. The model not only predicts the likelihood of default but also explains the factors contributing to each prediction.

## 📋 Table of Contents
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Details](#-model-details)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **Data Preprocessing**: Comprehensive data cleaning and feature engineering
- **Handling Class Imbalance**: Utilizes SMOTE (Synthetic Minority Over-sampling Technique)
- **Machine Learning Model**: Random Forest Classifier for robust predictions
- **Model Interpretation**: SHAP values for global and local explainability
- **Visualization**: Interactive and static plots for model insights
- **Performance Metrics**: Comprehensive evaluation metrics including precision, recall, F1-score, and AUC-ROC

## 🛠 Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Basic understanding of machine learning concepts

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/credit-card-default-prediction.git
   cd credit-card-default-prediction
   ```

2. **Create and activate a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

1. **Prepare your dataset**:
   - Place your credit card default dataset in the `data/` directory
   - The dataset should be in CSV format
   - Ensure the target variable is named 'default_payment_next_month' (or update the code accordingly)

2. **Run the prediction pipeline**:
   ```bash
   python credit_risk_prediction.py
   ```

3. **View the results**:
   - Model performance metrics will be displayed in the console
   - The following visualizations will be generated in the `results/` directory:
     - `confusion_matrix.png`: Model's confusion matrix
     - `roc_curve.png`: ROC curve and AUC score
     - `shap_summary.png`: Global feature importance
     - `shap_force_plot.png`: Local explanation for a sample prediction
     - `feature_importance.png`: Traditional feature importance

## 📁 Project Structure

```
credit-card-default-prediction/
├── data/                    # Directory for input data
│   └── credit_card_default.csv  # Example dataset
├── results/                 # Output directory for visualizations
├── credit_risk_prediction.py # Main script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🤖 Model Details

The model uses the following techniques:

- **Algorithm**: Random Forest Classifier
- **Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Feature Scaling**: StandardScaler
- **Model Interpretation**: SHAP values
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - AUC-ROC

## 📊 Results

After training, the model will display various performance metrics and generate the following visualizations:

1. **Confusion Matrix**: Shows true positives, false positives, true negatives, and false negatives
2. **ROC Curve**: Displays the trade-off between true positive rate and false positive rate
3. **SHAP Summary Plot**: Visualizes feature importance and impact on predictions
4. **Force Plot**: Explains individual predictions
5. **Feature Importance**: Traditional feature importance from the Random Forest model

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by Praveen Kumar Reddy Idamakanti
