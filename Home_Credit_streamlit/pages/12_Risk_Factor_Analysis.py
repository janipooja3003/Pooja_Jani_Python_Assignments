import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import load_application_train_data
from utils.filters import sidebar_filters, apply_filters

st.set_page_config(page_title="Risk Factor Analysis")
st.title("📈 Risk Factor Analysis")

try:
    df = load_application_train_data()
    filters = sidebar_filters(df)
    df_filtered = apply_filters(df, filters)

    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
    else:
        df_filtered = df_filtered.copy()
        df_filtered["AGE_YEARS"] = abs(df_filtered["DAYS_BIRTH"]) / 365.25
        df_filtered["EMPLOYMENT_YEARS"] = (
            df_filtered["DAYS_EMPLOYED"].replace(365243, pd.NA).abs() / 365.25
        )
        income = df_filtered["AMT_INCOME_TOTAL"].replace(0, pd.NA)
        df_filtered["CREDIT_TO_INCOME"] = df_filtered["AMT_CREDIT"] / income
        df_filtered["ANNUITY_TO_INCOME"] = df_filtered["AMT_ANNUITY"] / income
        df_filtered["Outcome"] = df_filtered["TARGET"].map({0: "Non-defaulted", 1: "Defaulted"})

        st.subheader("Risk Factor Overview")
        risk_cols = [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "AGE_YEARS",
            "EMPLOYMENT_YEARS",
            "CREDIT_TO_INCOME",
            "ANNUITY_TO_INCOME",
        ]
        st.dataframe(df_filtered[risk_cols].describe().T)

        st.subheader("External Score Risk")
        score_options = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        selected_score = st.selectbox("Select external score", score_options)
        score_data = df_filtered[[selected_score, "TARGET", "Outcome"]].dropna().copy()
        score_data["Score Band"] = pd.qcut(
            score_data[selected_score],
            q=5,
            labels=["Very Low", "Low", "Medium", "High", "Very High"],
            duplicates="drop",
        )
        score_summary = (
            score_data.groupby("Score Band", observed=True)["TARGET"]
            .agg(Applicants="count", Default_Rate="mean")
            .reset_index()
        )
        score_summary["Observed Default Rate %"] = score_summary["Default_Rate"] * 100
        score_fig = px.bar(
            score_summary,
            x="Score Band",
            y="Observed Default Rate %",
            title=f"Observed Default Rate by {selected_score} Band",
            color="Observed Default Rate %",
            color_continuous_scale="blues",
        )
        st.plotly_chart(score_fig, use_container_width=True)
        st.dataframe(score_summary[["Score Band", "Applicants", "Observed Default Rate %"]])

        st.subheader("Risk Factor Distributions by Outcome")
        factor_options = [
            "AGE_YEARS",
            "EMPLOYMENT_YEARS",
            "CREDIT_TO_INCOME",
            "ANNUITY_TO_INCOME",
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
        ]
        selected_factor = st.selectbox("Select a risk factor", factor_options)
        factor_fig = px.histogram(
            df_filtered,
            x=selected_factor,
            color="Outcome",
            nbins=40,
            barmode="overlay",
            opacity=0.65,
            title=f"{selected_factor} Distribution by Default Outcome",
            color_discrete_map={"Defaulted": "#ff7f0e", "Non-defaulted": "#2ca02c"},
        )
        factor_fig.update_layout(yaxis_title="Number of People")
        st.plotly_chart(factor_fig, use_container_width=True)

        st.subheader("Default Rate by Applicant Profile")

        def grouped_risk(data, column, label, title):
            grouped = (
            data.groupby(column, dropna=False)["TARGET"]
                .agg(Applicants="count", Default_Rate="mean")
                .reset_index()
                .rename(columns={column: label})
            )
            grouped["Observed Default Rate %"] = grouped["Default_Rate"] * 100
            figure = px.bar(
                grouped,
                x=label,
                y="Observed Default Rate %",
                title=title,
                color="Observed Default Rate %",
                color_continuous_scale = px.colors.qualitative.Pastel,
            )
            return grouped, figure

        profile_left, profile_right = st.columns(2)
        with profile_left:
            age_data = df_filtered.copy()
            age_data["Age Band"] = pd.cut(
                age_data["AGE_YEARS"],
                bins=[0, 25, 35, 45, 55, 65, float("inf")],
                labels=["Under 25", "25-35", "35-45", "45-55", "55-65", "65+"],
                right=False,
            )
            age_summary, age_fig = grouped_risk(
                age_data, "Age Band", "Age Band", "Default Rate by Age Band"
            )
            st.plotly_chart(age_fig, use_container_width=True)

            income_summary, income_fig = grouped_risk(
                df_filtered,
                "NAME_INCOME_TYPE", "Income Type", "Default Rate by Income Type"
            )
            st.plotly_chart(income_fig, use_container_width=True)

        with profile_right:
            children_data = df_filtered.copy()
            children_data["Children Status"] = children_data["CNT_CHILDREN"].gt(0).map(
                {True: "Has Children", False: "No Children"}
            )
            children_summary, children_fig = grouped_risk(
                children_data,
                "Children Status", "Children Status", "Default Rate by Children Status"
            )
            st.plotly_chart(children_fig, use_container_width=True)

            housing_summary, housing_fig = grouped_risk(
                df_filtered,
                "NAME_HOUSING_TYPE", "Housing Type", "Default Rate by Housing Type"
            )
            st.plotly_chart(housing_fig, use_container_width=True)

        st.subheader("Profile Risk Tables")
        st.dataframe(age_summary[["Age Band", "Applicants", "Observed Default Rate %"]])
        st.dataframe(income_summary[["Income Type", "Applicants", "Observed Default Rate %"]])
        st.dataframe(children_summary[["Children Status", "Applicants", "Observed Default Rate %"]])
        st.dataframe(housing_summary[["Housing Type", "Applicants", "Observed Default Rate %"]])

except FileNotFoundError:
    st.error("Dataset file not found. Add `data/application_train.csv`.")
