# Customer Churn Prediction

This project aims to predict whether a customer will churn based on service usage, contract details, and demographic information. It applies data preprocessing, feature engineering, and a Logistic Regression model to identify key patterns associated with customer churn.

## Dataset

- **File**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Source**: IBM Sample Data (available on Kaggle and other platforms)
- **Rows**: 7,043
- **Target**: `Churn` (Yes/No)

The dataset contains customer demographics, account information, and service usage details, which are used to train a churn prediction model.

## Objectives

- Explore and understand churn-related patterns
- Clean and preprocess the data
- Train and evaluate a Logistic Regression model
- Save the trained model for future use

## Technologies Used

- **Language**: Python  
- **Libraries**: pandas, NumPy, matplotlib, seaborn, scikit-learn, imbalanced-learn, joblib

## Workflow

### 1. Data Exploration and Visualization

- Reviewed feature distributions and types
- Visualized churn rates across different customer segments
- Identified correlations and trends relevant to churn

### 2. Data Preprocessing

- Handled missing or invalid values (e.g., in `TotalCharges`)
- Removed unnecessary features (e.g., `customerID`)
- Encoded categorical variables
- Scaled numerical features
- Checked for multicollinearity (using VIF)

### 3. Train-Test Preparation

- Split dataset into training and testing sets
- Applied SMOTE to balance classes in the training data

### 4. Modeling and Evaluation

- Trained a Logistic Regression model
- Evaluated performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Performed hyperparameter tuning (e.g., regularization strength)

### 5. Model Saving

- Saved the final trained model using `joblib` for reuse

## Results

Four variations of the Logistic Regression model were evaluated based on the use of SMOTE and hyperparameter tuning. The key performance metrics are summarized below:

| Model                | F2 Score | Recall | Precision | Accuracy | ROC AUC |
|---------------------|----------|--------|-----------|----------|---------|
| **Tuned_No_SMOTE**  | **0.7061** | **0.7941** | 0.4893    | 0.7249   | 0.8351  |
| Baseline_SMOTE      | 0.6554   | 0.6925 | 0.5396    | 0.7612   | 0.8264  |
| Tuned_SMOTE         | 0.6547   | 0.6925 | 0.5373    | 0.7598   | 0.8263  |
| Baseline_No_SMOTE   | 0.5860   | 0.5722 | **0.6485** | **0.8038** | **0.8359**  |

### Key Observations

- **Tuned_No_SMOTE** achieved the highest **F2 Score** and **Recall**, making it the best choice when prioritizing false negatives (e.g., retaining at-risk customers).
- **Baseline_No_SMOTE** had the highest **Precision**, **Accuracy**, and **ROC AUC**, suggesting better overall performance in balanced decisions.
- Applying **SMOTE** slightly improved recall but reduced precision, indicating potential overfitting or noise amplification in this context.
- Hyperparameter tuning (with `class_weight='balanced'`, regularization adjustments, etc.) improved recall without needing SMOTE.

> **Final model selected**: `Tuned_No_SMOTE` – balanced trade-off between recall and AUC, suitable for churn prediction where missing a churner is costly.


## How to Run

```bash
# Clone the repository
git clone https://github.com/quanho114/Predict-Customer-Churn.git
cd Predict-Customer-Churn

# Launch Jupyter Notebook
jupyter notebook Final_project.ipynb

