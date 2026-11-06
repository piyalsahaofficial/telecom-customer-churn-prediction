from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("model/churn_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Telecom churn prediction is running"}

@app.post("/predict")
def predict(customer : Customer):

    customer_data = {
        "gender": customer.gender,
        "SeniorCitizen": customer.SeniorCitizen,
        "Partner": customer.Partner,
        "Dependents": customer.Dependents,
        "tenure": customer.tenure,
        "PhoneService": customer.PhoneService,
        "MultipleLines": customer.MultipleLines,
        "InternetService": customer.InternetService,
        "OnlineSecurity": customer.OnlineSecurity,
        "OnlineBackup": customer.OnlineBackup,
        "DeviceProtection": customer.DeviceProtection,
        "TechSupport": customer.TechSupport,
        "StreamingTV": customer.StreamingTV,
        "StreamingMovies": customer.StreamingMovies,
        "Contract": customer.Contract,
        "PaperlessBilling": customer.PaperlessBilling,
        "PaymentMethod": customer.PaymentMethod,
        "MonthlyCharges": customer.MonthlyCharges,
        "TotalCharges": customer.TotalCharges
    }

    customer_df = pd.DataFrame([customer_data])
    customer_df = pd.get_dummies(customer_df, drop_first=True)
    customer_df = customer_df.reindex(columns=feature_columns, fill_value=0)
    prediction = model.predict(customer_df)
    probability = model.predict_proba(customer_df)
    churn_probability = round(probability[0][1]*100,2)
    
    

    if prediction[0] == 1:
        result = "Churn"
    else:
        result = "No Churn"

    return {
        "prediction": result,
        "churn_probability": str(churn_probability) + "%"
    }