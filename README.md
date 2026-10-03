# Telecom Customer Churn Prediction

## Overview

Machine learning project to predict whether a telecom customer will **stay or churn** using the **IBM Telco Customer Churn Dataset**.

## Dataset

**Dataset:** IBM Telco Customer Churn Dataset
**File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
**Original Size:** 7,043 rows × 21 columns
**Final Size:** 7,032 rows × 20 columns after cleaning

**Target:**

* `0` = Stayed
* `1` = Churned

## Workflow

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Encoding
* Train-Test Split
* Logistic Regression
* Random Forest
* Model Evaluation
* Confusion Matrix
* Feature Importance

## Results

| Model               | Accuracy | Churn Recall | Churn F1 |
| ------------------- | -------: | -----------: | -------: |
| Logistic Regression |   79.00% |          52% |      56% |
| Random Forest       |   78.54% |          48% |      54% |

Evaluation was performed on **1,407 test customers**.

## Top Random Forest Features

1. TotalCharges
2. MonthlyCharges
3. tenure
4. InternetService_Fiber optic
5. PaymentMethod_Electronic check

## Technologies

**Python · Pandas · NumPy · Matplotlib · Scikit-learn · Google Colab**

## Project Structure

```text
telecom-customer-churn-prediction/
├── Telecom_Customer_Churn_Prediction.ipynb
├── README.md
└── .gitignore
```

## Author

**Piyal Saha**
