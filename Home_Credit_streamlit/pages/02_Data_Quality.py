# 2. 🔍 Data Quality
# Question: "Can I trust this dataset?"
# Show:
# KPIs
# Rows
# Columns
# Duplicate Rows
# Total Missing Values
# % Missing Cells
# Charts
# Missing values by column
# Data types
# Unique values by categorical column
# Duplicate analysis
# You could have:
# st.metric("Rows", df.shape[0])
# st.metric("Columns", df.shape[1])
# st.metric("Duplicates", df.duplicated().sum())
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.feature_engineering import calc_kpis
from utils.preprocessing import load_application_train_data
from utils.filters import sidebar_filters, apply_filters
from utils.feature_engineering import calc_kpis #, top_bottom_summary
from utils.charts import line_chart, bar_chart, scatter_chart


st.set_page_config(page_title="Data Quality Analysis")

st.title("🔍 Data Quality Analysis")

try:
    df = load_application_train_data()
    filters = sidebar_filters(df)
    df_filtered = apply_filters(df, filters)

    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
    else:
        # KPIs
        st.subheader("Key Data Quality Metrics")
        cols = st.columns(5)
        cols[0].metric("Rows", f"{df_filtered.shape[0]:,}")
        cols[1].metric("Columns", f"{df_filtered.shape[1]:,}")
        cols[2].metric("Duplicate Rows", f"{df_filtered.duplicated().sum():,}")
        total_missing = df_filtered.isnull().sum().sum()
        total_missing_col = df_filtered.isnull().sum()
        total_missing_row = df_filtered.isnull().sum(axis=1)
        percent_missing = (total_missing / (df_filtered.shape[0] * df_filtered.shape[1])) * 100
        cols[3].metric("Total Missing Values", f"{total_missing:,}")
        cols[4].metric("% Missing Cells", f"{percent_missing:.2f}%")
        st.metric("Total Applicant With Missing Data", f"{len(total_missing_row[total_missing_row > 0])}")
        st.metric("Total Missing Columns", f"{len(total_missing_col[total_missing_col > 0])}")

        # Charts
        st.subheader("Missing Values by Column")
        missing_values = df_filtered.isnull().sum().reset_index()
        missing_values.columns = ["Column", "Missing Values"]
        missing_values = missing_values[missing_values["Missing Values"] > 0]
        if not missing_values.empty:
            fig = bar_chart(missing_values, group_col="Column", value_col="Missing Values", title="Missing Values by Column")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No missing values in the filtered dataset.")
    
    
            
except FileNotFoundError:
    st.error("Dataset file not found. Add `data/application_train.csv`.")


        

