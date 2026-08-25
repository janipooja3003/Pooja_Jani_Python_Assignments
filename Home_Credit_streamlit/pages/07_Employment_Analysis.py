import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.preprocessing import load_application_train_data
from utils.filters import sidebar_filters, apply_filters

st.set_page_config(page_title="Employment Analysis")
st.title("💼 Employment Analysis")

try:
    df = load_application_train_data()
    filters = sidebar_filters(df)
    df_filtered = apply_filters(df, filters)

    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
    else:
        df_filtered = df_filtered.copy()
        df_filtered["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        df_filtered["EMPLOYMENT_YEARS"] = abs(df_filtered["DAYS_EMPLOYED"]) / 365
        #df_filtered["DAYS_EMPLOYED"] = abs(df_filtered["DAYS_EMPLOYED"]) / 365

        st.subheader("Employment Summary")
        employment_summary = df_filtered[["DAYS_EMPLOYED", "EMPLOYMENT_YEARS", "OCCUPATION_TYPE", "NAME_INCOME_TYPE"]].describe(include="all").T
        st.dataframe(employment_summary)

        col1, col2 = st.columns(2)

        with col1:
            employment_hist = px.histogram(
                df_filtered,
                x="EMPLOYMENT_YEARS",
                nbins=40,
                title="Employment Duration Distribution",
                labels={"EMPLOYMENT_YEARS": "Years Employed"},
                color_discrete_sequence=["#2ca02c"],
            )
            employment_hist.update_layout(xaxis_title="Years Employed", yaxis_title="Number of People")
            st.plotly_chart(employment_hist, use_container_width=True)

            occ_count = (
                df_filtered["OCCUPATION_TYPE"].dropna().value_counts().reset_index()
            )
            occ_count.columns = ["Occupation", "Count"]
            occ_fig = px.bar(
                occ_count,
                x="Occupation",
                y="Count",
                title="Applicants by Occupation Type",
                color="Occupation",
            )
            st.plotly_chart(occ_fig, use_container_width=True)

        with col2:
            income_type_count = (
                df_filtered["NAME_INCOME_TYPE"].dropna().value_counts().reset_index()
            )
            income_type_count.columns = ["Income Type", "Count"]
            income_type_fig = px.bar(
                income_type_count,
                x="Income Type",
                y="Count",
                title="Applicants by Income Type",
                color="Income Type",
            )
            st.plotly_chart(income_type_fig, use_container_width=True)

            employment_by_income = (
                df_filtered.groupby("NAME_INCOME_TYPE", dropna=False)["EMPLOYMENT_YEARS"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            employment_by_income.columns = ["Income Type", "Average Employment Years"]
            emp_income_fig = px.bar(
                employment_by_income,
                x="Income Type",
                y="Average Employment Years",
                title="Average Employment Years by Income Type",
                color="Income Type",
            )
            st.plotly_chart(emp_income_fig, use_container_width=True)

        st.subheader("Default Rate by Employment Length")
        emp_default = df_filtered.copy()
        emp_default["Employment Band"] = pd.cut(
            emp_default["EMPLOYMENT_YEARS"],
            bins=[0, 1, 3, 5, 10, 20, float("inf")],
            labels=["< 1 year", "1-3 years", "3-5 years", "5-10 years", "10-20 years", "20+ years"],
            right=False,
        )

        emp_default_summary = (
            emp_default.groupby("Employment Band", dropna=False)["TARGET"]
            .agg(["count", "mean"])
            .reset_index()
        )
        emp_default_summary["Observed Default Rate %"] = emp_default_summary["mean"] * 100
        emp_default_summary = emp_default_summary.rename(columns={"count": "Applicants", "mean": "Observed Default Rate"})

        emp_default_fig = px.bar(
            emp_default_summary,
            x="Employment Band",
            y="Observed Default Rate %",
            title="Observed Default Rate by Employment Length",
            color="Employment Band",
        )
        st.plotly_chart(emp_default_fig, use_container_width=True)
        st.dataframe(emp_default_summary[["Employment Band", "Applicants", "Observed Default Rate", "Observed Default Rate %"]])

        st.subheader("Employment vs Default Risk")
        scatter_fig = px.scatter(
            df_filtered,
            x="EMPLOYMENT_YEARS",
            y="TARGET",
            title="Employment Years vs Default Risk",
            labels={"EMPLOYMENT_YEARS": "Years Employed", "TARGET": "Default Status"},
            opacity=0.5,
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

except FileNotFoundError:
    st.error("Dataset file not found. Add `data/application_train.csv`.")
