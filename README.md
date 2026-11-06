# Telecom Customer Churn Prediction

## Overview

A machine learning project that predicts whether a telecom customer is likely to stay or churn using the IBM Telco Customer Churn Dataset. The project includes an interactive Streamlit interface for customer churn prediction.

## Dataset

- **Dataset:** IBM Telco Customer Churn Dataset
- **Original Size:** 7,043 rows × 21 columns
- **Final Size:** 7,032 rows × 20 columns after cleaning
- **Target:** `Churn`
  - `0` = Stayed
  - `1` = Churned

## Workflow

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Encoding
- Train-Test Split
- Logistic Regression
- Random Forest
- Model Evaluation
- Confusion Matrix
- Feature Importance
- Model Saving and Deployment

## Model Performance

Evaluation on 1,407 test customers:

| Model | Accuracy | Churn Recall | Churn F1 |
|---|---:|---:|---:|
| Logistic Regression | 79.00% | 52% | 56% |
| Random Forest | 78.54% | 48% | 54% |

## Top Random Forest Features

1. TotalCharges
2. MonthlyCharges
3. tenure
4. InternetService_Fiber optic
5. PaymentMethod_Electronic check

## Technologies

Python · Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit · FastAPI · Joblib · Google Colab

## Project Structure

```text
telecom-customer-churn-prediction/
│
├── api/
│   ├── app.py
│   └── main.py
│
├── data/
│   └── telco_churn.csv
│
├── model/
│   ├── churn_model.pkl
│   └── feature_columns.pkl
│
├── Telecom_Customer_Churn_Prediction.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
cd api
python -m streamlit run app.py
```

Open your browser at:

`http://localhost:8501`

## Author

**Piyal Saha**