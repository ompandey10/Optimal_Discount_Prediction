import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Set page config
st.set_page_config(page_title="Optimal Discount Prediction", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))

@st.cache_data
def load_results():
    with open(os.path.join(MODELS_DIR, 'model_results.json'), 'r') as f:
        return json.load(f)

try:
    model = load_model()
    results = load_results()
except FileNotFoundError:
    st.error("Model or results not found. Please run the training script (`src/model_trainer.py`) first.")
    st.stop()

st.title("🏷️ Optimal Discount Prediction for Retail Products")
st.markdown("Predict expected sales based on product characteristics, discount levels, and market conditions.")

# Sidebar for Model Comparison
st.sidebar.header("📊 Model Comparison")
st.sidebar.markdown("Comparative study of regression algorithms based on test data:")

results_df = pd.DataFrame(results).T
st.sidebar.dataframe(results_df.style.highlight_max(subset=['R2'], color='green').highlight_min(subset=['RMSE', 'MAE', 'MSE'], color='green'))

st.sidebar.markdown("""
**Insights:**
- **R2 Score** indicates how well the model explains the variance in sales. Higher is better.
- **RMSE/MAE** represent prediction errors. Lower is better.
""")

st.header("🔮 Sales Prediction Tool")

col1, col2 = st.columns(2)
with col1:
    product_category = st.selectbox("Product Category", ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports'])
    original_price = st.number_input("Original Price ($)", min_value=1.0, value=100.0)
    proposed_discount = st.slider("Proposed Discount (%)", min_value=0, max_value=99, value=10)
    historical_sales = st.number_input("Historical Sales (units)", min_value=0.0, value=1000.0)

with col2:
    customer_demand = st.slider("Customer Demand Index (0.1 to 1.0)", min_value=0.1, max_value=1.0, value=0.5)
    season = st.selectbox("Season", ['Spring', 'Summer', 'Autumn', 'Winter'])
    promotion_status = st.selectbox("Promotion Status", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    competitor_price = st.number_input("Competitor Price ($)", min_value=1.0, value=100.0)

if st.button("Predict Expected Sales"):
    input_data = pd.DataFrame({
        'Product_Category': [product_category],
        'Original_Price': [original_price],
        'Proposed_Discount': [proposed_discount / 100.0],
        'Historical_Sales': [historical_sales],
        'Customer_Demand': [customer_demand],
        'Season': [season],
        'Promotion_Status': [promotion_status],
        'Competitor_Price': [competitor_price]
    })
    
    prediction = model.predict(input_data)[0]
    st.success(f"📈 Expected Sales: **{prediction:,.2f} units**")
    
    st.subheader("📊 Sales Projection Across Discount Levels")
    discount_range = np.arange(0, 75, 5) 
    projected_sales = []
    
    for d in discount_range:
        temp_input = input_data.copy()
        temp_input['Proposed_Discount'] = d / 100.0
        proj_sales = model.predict(temp_input)[0]
        projected_sales.append(proj_sales)
        
    projection_df = pd.DataFrame({
        'Discount (%)': discount_range,
        'Expected Sales (units)': projected_sales,
        'Estimated Revenue ($)': [s * (original_price * (1 - d/100.0)) for s, d in zip(projected_sales, discount_range)]
    })
    
    st.line_chart(projection_df.set_index('Discount (%)')[['Expected Sales (units)', 'Estimated Revenue ($)']])
    
    optimal_discount_idx = projection_df['Estimated Revenue ($)'].idxmax()
    optimal_discount = projection_df.loc[optimal_discount_idx, 'Discount (%)']
    max_revenue = projection_df.loc[optimal_discount_idx, 'Estimated Revenue ($)']
    
    st.info(f"💡 **Recommendation:** Based on revenue estimation, the optimal discount level is **{optimal_discount}%** resulting in a maximum revenue of **${max_revenue:,.2f}**.")
