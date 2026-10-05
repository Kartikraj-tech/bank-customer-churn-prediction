# Bank Customer Churn Prediction & Risk Scoring

A machine learning project for predicting bank customer churn and assigning customer-level risk scores. The project compares multiple classification models and provides an interactive Streamlit dashboard for churn analysis and what-if scenarios.

## Project Overview

The goal is to identify customers who may be likely to churn before they leave, using customer information such as age, credit score, account balance, number of products, activity status, geography, and salary.

### Target

`Exited`

- `0` → Customer retained
- `1` → Customer churned

Dataset size: **10,000 customers**

Churn distribution:

- Retained: **79.63%**
- Churned: **20.37%**

## Methodology

### Data Preprocessing

- Checked for missing values and duplicates
- Removed `CustomerId` and `Surname`
- Kept `Year` as provided in the dataset
- One-hot encoded `Geography` and `Gender`
- Applied feature scaling where required for Logistic Regression

### Feature Engineering

Created four additional features:

```text
BalanceSalaryRatio = Balance / EstimatedSalary
ProductDensity = NumOfProducts / (Tenure + 1)
EngagementProduct = IsActiveMember × NumOfProducts
AgeTenureInteraction = Age × Tenure
```

### Train-Test Strategy

Used an **80/20 stratified train-test split** with `random_state=42`.

Five-fold stratified cross-validation was also used to compare model performance.

## Models

The following models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

## Results

| Model | Accuracy | Precision | Recall | F1-Score | Test ROC-AUC | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 80.70% | 57.78% | 19.16% | 28.78% | 77.36% | 76.26% |
| Decision Tree | 78.30% | 46.82% | 48.89% | 47.84% | 67.35% | 68.35% |
| Random Forest | 86.55% | 78.05% | 47.17% | 58.81% | 84.90% | 84.66% |
| **Gradient Boosting** | **86.90%** | **78.66%** | 48.89% | **60.30%** | **86.82%** | **86.38%** |
| XGBoost | 85.25% | 69.44% | **49.14%** | 57.55% | 83.12% | 83.85% |

**Gradient Boosting** was selected as the final model because it achieved the strongest overall performance across the main evaluation metrics and the highest mean cross-validated ROC-AUC.

## Model Explainability

The project includes:

- Feature importance ranking
- SHAP value analysis
- Partial Dependence Plots (PDP)

The most important features in the final Gradient Boosting model were:

1. Age
2. Number of Products
3. Active Member status
4. Balance
5. Germany geography indicator
6. Engagement-Product interaction

## Risk Scoring

The final model produces a churn probability between `0` and `1`.

A binary churn flag uses a threshold of `0.50`:

```text
Probability >= 0.50 → Churn
Probability < 0.50  → Retained
```

Project-defined risk bands:

| Probability | Risk |
|---|---|
| 0–30% | Low |
| >30–60% | Medium |
| >60–80% | High |
| >80–100% | Very High |

## Streamlit Dashboard

The application includes:

### Customer Risk Calculator
Enter customer information and receive:

- Churn probability
- Risk level
- Binary churn prediction

### Churn Analytics
View:

- Customer churn statistics
- Churn probability distribution
- High-risk customer counts

### Feature Importance
Explore which features contribute most to the Gradient Boosting model.

### What-If Simulator
Change:

- Number of products
- Active membership

and compare the resulting predicted churn probability with the original prediction.

## Project Structure

```text
bank-customer-churn-prediction/
│
├── European_Bank.csv
├── bank_churn_model.pkl
├── data.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- XGBoost
- SHAP
- Streamlit
- Joblib
- Jupyter Notebook


## Limitations

- The dataset contains only one year (`2025`), so temporal churn trends cannot be studied.
- Model predictions represent statistical patterns and should not be interpreted as causal effects.
- Customers with three or four products are relatively few, so their very high observed churn rates should be interpreted cautiously.
- Risk thresholds are project-defined and would need validation and calibration before production use.

## Disclaimer

This project is intended for educational and demonstration purposes. The predictions should be treated as decision-support information and not as the sole basis for real-world customer decisions.

