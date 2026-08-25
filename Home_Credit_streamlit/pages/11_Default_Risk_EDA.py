import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import load_application_train_data
from utils.filters import sidebar_filters, apply_filters

st.set_page_config(page_title="Default Risk EDA")
st.title("⚠️ Default Risk EDA")

try:
    df = load_application_train_data()
    filters = sidebar_filters(df)
    df_filtered = apply_filters(df, filters)

    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
    else:
        df_filtered = df_filtered.copy()
        df_filtered["Outcome"] = df_filtered["TARGET"].map({0: "Non-defaulted", 1: "Defaulted"})
        df_filtered["AGE_YEARS"] = abs(df_filtered["DAYS_BIRTH"]) / 365.25
        df_filtered["EMPLOYMENT_YEARS"] = (
            df_filtered["DAYS_EMPLOYED"].replace(365243, pd.NA).abs() / 365.25
        )

        default_count = int((df_filtered["TARGET"] == 1).sum())
        non_default_count = int((df_filtered["TARGET"] == 0).sum())
        default_rate = df_filtered["TARGET"].mean() * 100

        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("Total Applicants", f"{len(df_filtered):,}")
        metric_2.metric("Defaulted Applicants", f"{default_count:,}")
        metric_3.metric("Observed Default Rate", f"{default_rate:.2f}%")

        outcome_data = (
            df_filtered["Outcome"]
            .value_counts()
            .rename_axis("Outcome")
            .reset_index(name="Applicants")
        )
        outcome_fig = px.bar(
            outcome_data,
            x="Outcome",
            y="Applicants",
            title="Defaulted vs Non-defaulted Applicants",
            color="Outcome",
            color_discrete_map={"Defaulted": "#d62728", "Non-defaulted": "#2ca02c"},
        )
        st.plotly_chart(outcome_fig, use_container_width=True)

        def risk_by_group(column, label, title):
            result = (
                df_filtered.groupby(column, dropna=False)["TARGET"]
                .agg(Applicants="count", Default_Rate="mean")
                .reset_index()
                .rename(columns={column: label})
            )
            result["Observed Default Rate %"] = result["Default_Rate"] * 100
            figure = px.bar(
                result,
                x=label,
                y="Observed Default Rate %",
                title=title,
                color="Observed Default Rate %",
                color_continuous_scale="Reds",
            )
            return result, figure

        left, right = st.columns(2)
        with left:
            gender_summary, gender_fig = risk_by_group(
                "CODE_GENDER", "Gender", "Default Rate by Gender"
            )
            st.plotly_chart(gender_fig, use_container_width=True)

            income_summary, income_fig = risk_by_group(
                "NAME_INCOME_TYPE", "Income Type", "Default Rate by Income Type"
            )
            goods_by_income = (
                df_filtered.assign(
                    GOODS_PRICE_TO_INCOME=(
                        df_filtered["AMT_GOODS_PRICE"]
                        / df_filtered["AMT_INCOME_TOTAL"].replace(0, pd.NA)
                    )
                )
                .groupby("NAME_INCOME_TYPE", dropna=False)["GOODS_PRICE_TO_INCOME"]
                .mean()
                .mul(100)
                .reset_index(name="Average Goods Price / Income %")
                .rename(columns={"NAME_INCOME_TYPE": "Income Type"})
            )
            income_summary = income_summary.merge(goods_by_income, on="Income Type", how="left")
            st.plotly_chart(income_fig, use_container_width=True)

        with right:
            education_summary, education_fig = risk_by_group(
                "NAME_EDUCATION_TYPE", "Education Type", "Default Rate by Education"
            )
            st.plotly_chart(education_fig, use_container_width=True)

            contract_summary, contract_fig = risk_by_group(
                "NAME_CONTRACT_TYPE", "Contract Type", "Default Rate by Contract Type"
            )
            st.plotly_chart(contract_fig, use_container_width=True)

        st.subheader("Numeric Risk Factor Distributions")
        numeric_options = ["AGE_YEARS", "EMPLOYMENT_YEARS", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]
        selected_numeric = st.selectbox("Select a risk factor", numeric_options)
        distribution_fig = px.histogram(
            df_filtered,
            x=selected_numeric,
            color="Outcome",
            nbins=40,
            barmode="overlay",
            opacity=0.65,
            title=f"{selected_numeric} Distribution by Default Outcome",
            color_discrete_map={"Defaulted": "#d62728", "Non-defaulted": "#2ca02c"},
        )
        distribution_fig.update_layout(yaxis_title="Number of People")
        st.plotly_chart(distribution_fig, use_container_width=True)

        st.subheader("Default Risk Tables")
        table_left, table_right = st.columns(2)
        with table_left:
            st.write("Default Rate by Gender")
            st.dataframe(gender_summary[["Gender", "Applicants", "Observed Default Rate %"]])
            st.write("Default Rate by Income Type")
            st.dataframe(
                income_summary[
                    [
                        "Income Type",
                        "Applicants",
                        "Average Goods Price / Income %",
                        "Observed Default Rate %",
                    ]
                ]
            )
        with table_right:
            st.write("Default Rate by Education")
            st.dataframe(education_summary[["Education Type", "Applicants", "Observed Default Rate %"]])
            st.write("Default Rate by Contract Type")
            st.dataframe(contract_summary[["Contract Type", "Applicants", "Observed Default Rate %"]])

except FileNotFoundError:
    st.error("Dataset file not found. Add `data/application_train.csv`.")
