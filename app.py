# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:10:50 2026

@author: Sanu
"""

import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🛒 Product Clustering App")

st.sidebar.header("Enter Product Details")

sales_per_month = st.sidebar.number_input("Sales per Month")
cv_sales = st.sidebar.number_input("CV Sales")
retail_ratio = st.sidebar.number_input("Retail Ratio")
warehouse_ratio = st.sidebar.number_input("Warehouse Ratio")
active_months = st.sidebar.number_input("Active Months")

if st.button("Predict"):

    data = pd.DataFrame([[sales_per_month, cv_sales,
                          retail_ratio, warehouse_ratio, active_months]],
                        columns=[
                            'sales_per_month','cv_sales',
                            'retail_ratio','warehouse_ratio','active_months'
                        ])

    scaled = scaler.transform(data)
    cluster = model.predict(scaled)[0]

    if cluster == 0:
        st.success("Low / Moderate Demand Product")
    else:
        st.success("High Demand Product")