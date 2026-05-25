# CreditWise Loan Approval System

> End-to-end supervised machine learning pipeline to predict loan approval decisions.

## Overview
Built a binary classification system analyzing **1,000 loan records across 20 features** 
to automate loan approval predictions. Compared three ML algorithms and identified key 
financial drivers of loan approval through EDA.

## Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | **86.5%** | 80.35% | 73.77% | 76.92% |
| Logistic Regression | X% | X% | X% | X% |
| KNN | X% | X% | X% | X% |

## Key Findings
- **Credit Score** = strongest positive predictor of loan approval
- **DTI Ratio** = strongest negative predictor (higher debt = lower approval)
- Correlation heatmap confirmed both findings statistically

## Tech Stack
`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn` `Jupyter Notebook`

## ML Pipeline
1. Exploratory Data Analysis (EDA) + correlation heatmap
2. Data preprocessing — imputation, encoding, scaling
3. Model training — KNN, Logistic Regression, Naive Bayes
4. Evaluation — Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook creditwise_loan_approval.ipynb
```

## Dataset
1,000 loan applications | 20 features | Binary target: Approved / Rejected
