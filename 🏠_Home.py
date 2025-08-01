# 🏠_Home.py
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="🏠",
    layout="centered"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
model = joblib.load('models/churn_model_logistic_regression.pkl')
    return model

model = load_model()

# --- APP LAYOUT ---
st.title("👤 Customer Service - Churn Risk Checker")
st.write("Enter a customer's details to get an instant churn probability score.")

# --- INPUT FORM ---
with st.form("churn_prediction_form"):
    st.subheader("Customer Account Information")
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        contract = st.selectbox("Contract Type", ('Month-to-month', 'One year', 'Two year'))
        monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0)

    with col2:
        internet_service = st.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
        tech_support = st.selectbox("Tech Support", ('Yes', 'No', 'No internet service'))
        total_charges = tenure * monthly_charges # A simple approximation

    submitted = st.form_submit_button("Check Churn Risk")

# --- PREDICTION LOGIC ---
if submitted:
    # Create a single-row DataFrame from the inputs
    # The column names MUST match the training data columns
    input_data = pd.DataFrame({
        'gender': ['Male'], 'SeniorCitizen': [0], 'Partner': ['Yes'], 'Dependents': ['No'],
        'tenure': [tenure], 'PhoneService': ['Yes'], 'MultipleLines': ['No'],
        'InternetService': [internet_service], 'OnlineSecurity': ['No'], 'OnlineBackup': ['Yes'],
        'DeviceProtection': ['No'], 'TechSupport': [tech_support], 'StreamingTV': ['No'],
        'StreamingMovies': ['No'], 'Contract': [contract], 'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'], 'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # Get prediction probability
    churn_probability = model.predict_proba(input_data)[:, 1][0]

    # Display the result
    st.subheader("Churn Prediction Result")
    
    if churn_probability < 0.3:
        st.success(f"LOW RISK (Probability: {churn_probability:.2%})")
        st.write("This customer is likely to stay. Focus on standard service.")
    elif churn_probability < 0.6:
        st.warning(f"MEDIUM RISK (Probability: {churn_probability:.2%})")
        st.write("This customer is a potential churn risk. Ensure excellent service and address all concerns.")
    else:
        st.error(f"HIGH RISK (Probability: {churn_probability:.2%})")
        st.write("Immediate action recommended! Consider offering a loyalty discount or plan upgrade.")
