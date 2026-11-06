
import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="centered"
)

# Locate project root and model files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")

# Load trained model
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

# Interface
st.title("📡 Telecom Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to leave.")

st.divider()

st.subheader("Customer Information")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", 0, 72, 12)

        phone = st.selectbox("Phone Service", ["Yes", "No"])

        multiple = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

        backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

    with col2:
        protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

        support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

        tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

        movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
            "Monthly Charges ($)", min_value=0.0, value=70.0
        )

        total = st.number_input(
            "Total Charges ($)", min_value=0.0, value=840.0
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )

# Prediction
if submitted:

    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    customer_df = pd.DataFrame([customer])

    customer_df = pd.get_dummies(
        customer_df,
        drop_first=True
    )

    customer_df = customer_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(customer_df)[0]

    probabilities = model.predict_proba(customer_df)[0]

    churn_index = list(model.classes_).index(1)

    churn_probability = probabilities[churn_index] * 100

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

    st.metric(
        "Churn Probability",
        f"{churn_probability:.2f}%"
    )

    st.progress(float(churn_probability / 100))

    st.caption(
        "Prediction is based on the trained machine learning model."
    )
