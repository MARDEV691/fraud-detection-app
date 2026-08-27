import streamlit as st
import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier

st.set_page_config(
    page_title="E-Commerce Fraud Detection Portal",
    page_icon="🛡️",
    layout="wide"
)

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = XGBClassifier()
    model.load_model("xgboost_fraud_model.json")
    
    with open("model_config.json", "r") as f:
        config = json.load(f)
    threshold = config.get("optimal_threshold", 0.4759)
    
    with open("model_features.json", "r") as f:
        expected_features = json.load(f)
        
    with open("scaler_params.json", "r") as f:
        scaler_params = json.load(f)
        
    return model, threshold, expected_features, scaler_params

model, optimal_threshold, expected_features, scaler_params = load_artifacts()

st.title("🛡️ E-Commerce Real-Time Fraud Detection System")
st.markdown("Enter transaction details below to evaluate the fraud risk score using our optimized XGBoost classification engine.")
st.markdown("---")

# User Input Form
st.subheader("📋 Transaction Details")
col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=20000.0, value=35.0, step=5.0)
    quantity = st.number_input("Item Quantity", min_value=1, max_value=50, value=1, step=1)
    payment_method = st.selectbox("Payment Method", ["credit card", "debit card", "paypal", "bank transfer"])

with col2:
    customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=42, step=1)
    account_age_days = st.number_input("Account Age (Days)", min_value=0, max_value=2000, value=350, step=1)
    product_category = st.selectbox("Product Category", ["clothing", "electronics", "home & garden", "toys & games", "health & beauty"])

with col3:
    device_used = st.selectbox("Device Used", ["desktop", "mobile", "tablet"])
    transaction_hour = st.slider("Transaction Hour (0 - 23)", min_value=0, max_value=23, value=14)
    day_of_week = st.selectbox("Day of Week", ["Wednesday", "Monday", "Tuesday", "Thursday", "Friday", "Saturday", "Sunday"])
    month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
    address_mismatch = st.checkbox("Shipping Address differs from Billing Address")

st.markdown("---")

if st.button("🚀 Analyze Transaction Risk", use_container_width=True):
    # 1. Feature Engineering
    unit_price = amount / max(quantity, 1)
    is_late_night = 1 if (transaction_hour < 6 or transaction_hour >= 22) else 0
    
    device_freq_map = {'desktop': 7500, 'mobile': 7500, 'tablet': 3907}
    device_freq = device_freq_map.get(device_used, 5000)
    
    amount_to_age = amount / (account_age_days + 1)
    amount_above_median = 1 if amount > 150.0 else 0

    raw_dict = {
        'Transaction Amount': amount,
        'Quantity': quantity,
        'Customer Age': customer_age,
        'Account Age Days': account_age_days,
        'Transaction Hour': transaction_hour,
        'Address_Mismatch': int(address_mismatch),
        'Unit_Price': unit_price,
        'Is_Late_Night': is_late_night,
        'Device_Frequency': device_freq,
        'Amount_to_AccountAge_Ratio': amount_to_age,
        'Amount_Above_Median': amount_above_median
    }

    # 2. Apply Exact Standardization (z = (x - mean) / scale)
    scaled_dict = {}
    mean_map = dict(zip(scaler_params['num_cols'], scaler_params['mean']))
    scale_map = dict(zip(scaler_params['num_cols'], scaler_params['scale']))

    for col, val in raw_dict.items():
        if col in mean_map:
            scaled_dict[f"num__{col}"] = (val - mean_map[col]) / scale_map[col]

    # 3. Apply One-Hot Encoding
    processed_row = {feat: 0.0 for feat in expected_features}
    
    # Fill in scaled numerical values
    for feat, val in scaled_dict.items():
        if feat in processed_row:
            processed_row[feat] = val

    # Fill in one-hot categorical values
    cat_mappings = {
        'Payment Method': payment_method,
        'Product Category': product_category,
        'Device Used': device_used,
        'Day of Week': day_of_week,
        'Month': month
    }
    for cat_col, val in cat_mappings.items():
        target_feat = f"cat__{cat_col}_{val}"
        if target_feat in processed_row:
            processed_row[target_feat] = 1.0

    final_input_df = pd.DataFrame([processed_row])[expected_features]

    # 4. Predict
    fraud_probability = model.predict_proba(final_input_df.values)[0, 1]

    # 5. Display Outcome
    st.subheader("📊 Assessment Outcome")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric(
            label="Calculated Fraud Probability",
            value=f"{fraud_probability * 100:.2f}%"
        )
        st.caption(f"Decision Threshold: {optimal_threshold:.4f}")

    with res_col2:
        if fraud_probability >= optimal_threshold:
            st.error("🚨 **HIGH RISK TRANSACTION FLAGGED**")
            st.write("This transaction exceeds the risk threshold and is recommended for manual review.")
        else:
            st.success("✅ **TRANSACTION APPROVED**")
            st.write("Risk level is within normal parameters. Proceed with payment authorization.")