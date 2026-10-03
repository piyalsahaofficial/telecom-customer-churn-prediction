# Telecom Customer Churn Prediction

## Project Overview

This project predicts whether a telecom customer is likely to leave the company (churn) based on customer information and service details.

The project uses the IBM Telco Customer Churn dataset and follows a complete beginner-friendly machine learning workflow.

## Objectives

* Understand the telecom customer dataset
* Clean and prepare the data
* Explore customer churn patterns
* Convert categorical data into numerical features
* Train machine learning models
* Evaluate model performance
* Identify important features related to model predictions

## Dataset

The dataset contains information about telecom customers, including:

* Customer tenure
* Monthly charges
* Total charges
* Contract type
* Internet service
* Payment method
* Online security
* Technical support
* Senior citizen status
* Churn status

### Target Variable

* `0` = Customer stayed
* `1` = Customer churned

## Machine Learning Workflow

1. Data Collection
2. Data Understanding
3. Data Cleaning
4. Exploratory Data Analysis (EDA)
5. Feature Preparation
6. Train-Test Split
7. Logistic Regression
8. Random Forest
9. Model Evaluation
10. Feature Importance Analysis

## Models Used

### 1. Logistic Regression

Logistic Regression was used as a baseline classification model for predicting customer churn.

### 2. Random Forest

Random Forest was used as a second classification model using multiple decision trees.

## Results

| Model               | Accuracy | Churn Precision | Churn Recall | Churn F1-Score |
| ------------------- | -------: | --------------: | -----------: | -------------: |
| Logistic Regression |   79.00% |             62% |          52% |            56% |
| Random Forest       |   78.54% |             63% |          48% |            54% |

The models were evaluated on 1,407 test customers.

## Feature Importance

The Random Forest model identified the following as its top features:

1. TotalCharges
2. MonthlyCharges
3. Tenure
4. InternetService_Fiber optic
5. PaymentMethod_Electronic check
6. OnlineSecurity_Yes
7. Contract_Two year
8. gender_Male
9. TechSupport_Yes
10. PaperlessBilling_Yes

`TotalCharges`, `MonthlyCharges`, and `tenure` had the highest feature-importance values in the Random Forest model.

Feature importance describes how useful features were to the model's predictions; it does not establish that a feature causes customer churn.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Google Colab
* Jupyter Notebook

## Project Structure

```text
telecom-customer-churn-prediction/
│
├── Telecom_Customer_Churn_Prediction.ipynb
├── README.md
└── .gitignore
```

## How to Run

1. Download or clone this repository.
2. Open `Telecom_Customer_Churn_Prediction.ipynb`.
3. Open the notebook in Google Colab or Jupyter Notebook.
4. Upload the dataset when prompted.
5. Run the cells in order.

## Learning Outcomes

Through this project, I practiced:

* Data cleaning
* Exploratory data analysis
* Categorical feature encoding
* Train-test splitting
* Classification models
* Model evaluation
* Confusion matrix analysis
* Feature importance analysis
* Machine learning workflow development

## Author

Piyal Saha

Computer Science & Engineering
