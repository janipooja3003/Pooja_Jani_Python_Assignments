import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import load_application_train_data
from utils.filters import sidebar_filters, apply_filters

st.set_page_config(page_title="Loan Application Analysis")
st.title("💳 Loan Application Analysis")

try:
    df = load_application_train_data()
    filters = sidebar_filters(df)
    df_filtered = apply_filters(df, filters)

    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
    else:
        df_filtered = df_filtered.copy()
        df_filtered["CREDIT_TO_INCOME"] = (
            df_filtered["AMT_CREDIT"] / df_filtered["AMT_INCOME_TOTAL"].replace(0, pd.NA)
        )
        df_filtered["ANNUITY_TO_INCOME"] = (
            df_filtered["AMT_ANNUITY"] / df_filtered["AMT_INCOME_TOTAL"].replace(0, pd.NA)
        )

        st.subheader("Loan Application Summary")
        summary_cols = [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "CREDIT_TO_INCOME",
            "ANNUITY_TO_INCOME",
        ]
        summary = df_filtered[summary_cols].describe().T
        summary.index = [
            "Annual Income",
            "Credit Amount",
            "Annuity",
            "Goods Price",
            "Credit / Income",
            "Annuity / Income",
        ]
        st.dataframe(summary)

        left, right = st.columns(2)
        with left:
            contract_data = (
                df_filtered["NAME_CONTRACT_TYPE"]
                .value_counts()
                .rename_axis("Contract Type")
                .reset_index(name="Applicants")
            )
            contract_fig = px.bar(
                contract_data,
                x="Contract Type",
                y="Applicants",
                title="Applications by Contract Type",
                color="Contract Type",
            )
            st.plotly_chart(contract_fig, use_container_width=True)

            credit_fig = px.histogram(
                df_filtered,
                x="AMT_CREDIT",
                nbins=40,
                title="Requested Credit Distribution",
                labels={"AMT_CREDIT": "Credit Amount"},
                color_discrete_sequence=["#1f77b4"],
            )
            credit_fig.update_layout(yaxis_title="Number of People")
            st.plotly_chart(credit_fig, use_container_width=True)

        with right:
            annuity_fig = px.histogram(
                df_filtered,
                x="AMT_ANNUITY",
                nbins=40,
                title="Annuity Distribution",
                labels={"AMT_ANNUITY": "Annuity Amount"},
                color_discrete_sequence=["#ff7f0e"],
            )
            annuity_fig.update_layout(yaxis_title="Number of People")
            st.plotly_chart(annuity_fig, use_container_width=True)

            goods_fig = px.scatter(
                df_filtered,
                x="AMT_GOODS_PRICE",
                y="AMT_CREDIT",
                title="Goods Price vs Credit Amount",
                labels={
                    "AMT_GOODS_PRICE": "Goods Price",
                    "AMT_CREDIT": "Credit Amount",
                },
                opacity=0.5,
            )
            st.plotly_chart(goods_fig, use_container_width=True)

        st.subheader("Default Rate by Credit Amount")
        loan_bands = df_filtered.copy()
        loan_bands["Credit Band"] = pd.qcut(
            loan_bands["AMT_CREDIT"],
            q=5,
            labels=["Lowest", "Low", "Middle", "High", "Highest"],
            duplicates="drop",
        )
        band_summary = (
            loan_bands.groupby("Credit Band", observed=True)["TARGET"]
            .agg(Applicants="count", Default_Rate="mean")
            .reset_index()
        )
        band_summary["Observed Default Rate %"] = band_summary["Default_Rate"] * 100
        band_fig = px.bar(
            band_summary,
            x="Credit Band",
            y="Observed Default Rate %",
            title="Observed Default Rate by Credit Amount Band",
            color="Observed Default Rate %",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(band_fig, use_container_width=True)
        st.dataframe(band_summary[["Credit Band", "Applicants", "Observed Default Rate %"]])

        st.subheader("Default Rate by Contract Type")
        contract_default = (
            df_filtered.groupby("NAME_CONTRACT_TYPE")["TARGET"]
            .agg(Applicants="count", Default_Rate="mean")
            .reset_index()
            .rename(columns={"NAME_CONTRACT_TYPE": "Contract Type"})
        )
        contract_default["Observed Default Rate %"] = contract_default["Default_Rate"] * 100
        contract_default_fig = px.bar(
            contract_default,
            x="Contract Type",
            y="Observed Default Rate %",
            title="Observed Default Rate by Contract Type",
            color="Contract Type",
        )
        st.plotly_chart(contract_default_fig, use_container_width=True)
        st.dataframe(contract_default[["Contract Type", "Applicants", "Observed Default Rate %"]])

except FileNotFoundError:
    st.error("Dataset file not found. Add `data/application_train.csv`.")
