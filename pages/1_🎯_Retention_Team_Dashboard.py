# pages/1_🎯_Retention_Team_Dashboard.py
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Retention Team Dashboard",
    page_icon="🎯",
    layout="wide"
)

# --- MODEL AND DATA LOADING ---
@st.cache_resource
def load_model():
    """Load the trained model pipeline."""
    model = joblib.load('churn_model_logistic_regression.pkl')
    return model

@st.cache_data
def load_data():
    """Load the full customer dataset from the CSV in the repository."""
    # This path is relative to the root of your GitHub repository
    df = pd.read_csv('data/telco_customer_data.csv')
    return df

model = load_model()
df_full = load_data()

# --- APP LAYOUT ---
st.title("🎯 Retention Team - Daily Action Dashboard")
st.write(
    "This dashboard identifies customers with the highest churn risk, "
    "allowing the retention team to prioritize their outreach efforts."
)

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Customers")

# Probability slider
prob_threshold = st.sidebar.slider(
    "Show customers with churn probability above:",
    min_value=0.0,
    max_value=1.0,
    value=0.5, # Default to showing customers with > 50% churn risk
    step=0.05
)

# Contract type multi-select
contract_filter = st.sidebar.multiselect(
    "Filter by Contract Type:",
    options=df_full['Contract'].unique(),
    default=df_full['Contract'].unique()
)

# --- PREDICTION AND DISPLAY LOGIC ---
# Create a copy to avoid modifying the cached dataframe
df_processed = df_full.copy()

# Drop the original 'Churn' column if it exists, as we will predict it
if 'Churn' in df_processed.columns:
    df_processed = df_processed.drop('Churn', axis=1)

# Get churn probabilities for the entire dataset
churn_probabilities = model.predict_proba(df_processed)[:, 1]

# Add the probabilities to the dataframe
df_processed['ChurnProbability'] = churn_probabilities

# --- FILTERING DATA BASED ON USER INPUT ---
# Filter by churn probability
df_filtered = df_processed[df_processed['ChurnProbability'] >= prob_threshold]

# Filter by contract type
df_filtered = df_filtered[df_filtered['Contract'].isin(contract_filter)]

# Sort by churn probability to prioritize the highest-risk customers
df_sorted = df_filtered.sort_values(by='ChurnProbability', ascending=False)

# --- DISPLAY RESULTS ---
st.header("Prioritized Customer Call List")
st.write(f"Found **{len(df_sorted)}** customers matching the criteria.")

# Display the interactive dataframe
st.dataframe(df_sorted[[
    'customerID',
    'ChurnProbability',
    'tenure',
    'Contract',
    'MonthlyCharges',
    'TotalCharges'
]].style.format({'ChurnProbability': '{:.2%}', 'MonthlyCharges': '${:.2f}', 'TotalCharges': '${:.2f}'}),
use_container_width=True)

# --- DOWNLOAD BUTTON ---
@st.cache_data
def convert_df_to_csv(df):
    """Convert DataFrame to CSV for downloading."""
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df_sorted)

st.download_button(
    label="Download Call List as CSV",
    data=csv,
    file_name=f'retention_call_list_{prob_threshold:.0%}_risk.csv',
    mime='text/csv',
)
