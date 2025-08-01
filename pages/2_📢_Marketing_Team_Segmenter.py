# pages/2_📢_Marketing_Team_Segmenter.py
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Marketing Campaign Segmenter",
    page_icon="📢",
    layout="wide"
)

# --- MODEL AND DATA LOADING ---
# Use st.cache_resource for resource-heavy objects that don't change (like models)
@st.cache_resource
def load_model():
    """Load the trained model pipeline."""
    model = joblib.load('models/churn_model_logistic_regression.pkl')
    return model

# Use st.cache_data for data that can be loaded and processed
@st.cache_data
def load_and_predict_data():
    """Load the full customer dataset and append churn probabilities."""
    # This path is relative to the root of your GitHub repository
    df = pd.read_csv('data/telco_customer_data.csv')
    
    # Create a copy for predictions to avoid modifying the cached original
    df_predict = df.copy()
    if 'Churn' in df_predict.columns:
        df_predict = df_predict.drop('Churn', axis=1)

    # Get churn probabilities for the entire dataset
    churn_probabilities = model.predict_proba(df_predict)[:, 1]
    
    # Add the probabilities back to the original dataframe
    df['ChurnProbability'] = churn_probabilities
    return df

model = load_model()
df_with_preds = load_and_predict_data()

# --- APP LAYOUT ---
st.title("📢 Marketing Campaign Audience Builder")
st.write(
    "Use the filters in the sidebar to build a targeted customer segment "
    "for your next marketing campaign."
)

# --- SIDEBAR FILTERS ---
st.sidebar.header("Build Your Segment")

# Filter by Churn Probability Range
prob_range = st.sidebar.slider(
    "Filter by Churn Probability Range:",
    min_value=0.0,
    max_value=1.0,
    value=(0.4, 0.7), # Default range
    step=0.05
)

# Filter by Customer Tenure
tenure_range = st.sidebar.slider(
    "Filter by Customer Tenure (Months):",
    min_value=0,
    max_value=72,
    value=(0, 12), # Default to newer customers
    step=1
)

# Filter by Contract Type
contract_filter = st.sidebar.multiselect(
    "Filter by Contract Type:",
    options=df_with_preds['Contract'].unique(),
    default=['Month-to-month'] # Default to the highest churn risk group
)

# Filter by Internet Service
internet_filter = st.sidebar.multiselect(
    "Filter by Internet Service:",
    options=df_with_preds['InternetService'].unique(),
    default=df_with_preds['InternetService'].unique()
)


# --- APPLYING FILTERS ---
# Start with the full dataset containing predictions
df_filtered = df_with_preds

# Apply each filter sequentially
df_filtered = df_filtered[
    (df_filtered['ChurnProbability'] >= prob_range[0]) &
    (df_filtered['ChurnProbability'] <= prob_range[1])
]
df_filtered = df_filtered[
    (df_filtered['tenure'] >= tenure_range[0]) &
    (df_filtered['tenure'] <= tenure_range[1])
]
df_filtered = df_filtered[df_filtered['Contract'].isin(contract_filter)]
df_filtered = df_filtered[df_filtered['InternetService'].isin(internet_filter)]


# --- DISPLAY RESULTS ---
st.header("Generated Customer Segment")

# Show a metric with the number of customers in the segment
st.metric(label="Total Customers in Segment", value=f"{len(df_filtered)}")

st.write("This table contains the customers who match all the selected criteria.")

# Display the interactive dataframe with key marketing info
st.dataframe(df_filtered[[
    'customerID',
    'ChurnProbability',
    'tenure',
    'Contract',
    'InternetService',
    'MonthlyCharges'
]].style.format({'ChurnProbability': '{:.2%}', 'MonthlyCharges': '${:.2f}'}),
use_container_width=True)

# --- DOWNLOAD BUTTON ---
# A function to convert the dataframe to CSV, cached for performance
@st.cache_data
def convert_df_to_csv(df):
    """Convert DataFrame to CSV for downloading."""
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df_filtered)

# The download button itself
st.download_button(
   label="Download Segment as CSV",
   data=csv,
   file_name='marketing_segment.csv',
   mime='text/csv',
)
