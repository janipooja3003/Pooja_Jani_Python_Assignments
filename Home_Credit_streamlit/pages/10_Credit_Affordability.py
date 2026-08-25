import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import load_application_train_data
from utils.filters import sidebar_filters, apply_filters

st.set_page_config(page_title="Credit Affordability")
st.title("📊 Credit Affordability")

try:
    df = load_application_train_data()
    filters = sidebar_filters(df)
    df_filtered = apply_filters(df, filters)

    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
    else:
        df_filtered = df_filtered.copy()
        income = df_filtered["AMT_INCOME_TOTAL"].replace(0, pd.NA)
        df_filtered["CREDIT_TO_INCOME"] = df_filtered["AMT_CREDIT"] / income
        df_filtered["ANNUITY_TO_INCOME"] = df_filtered["AMT_ANNUITY"] / income
        df_filtered["CREDIT_TO_GOODS_PRICE"] = (
            df_filtered["AMT_CREDIT"] / df_filtered["AMT_GOODS_PRICE"].replace(0, pd.NA)
        )

        st.subheader("Affordability Summary")
        metric_cols = ["CREDIT_TO_INCOME", "ANNUITY_TO_INCOME", "CREDIT_TO_GOODS_PRICE"]
        summary = df_filtered[metric_cols].describe().T
        summary.index = ["Credit / Income", "Annuity / Income", "Credit / Goods Price"]
        st.dataframe(summary)

        average_credit_ratio = df_filtered["CREDIT_TO_INCOME"].mean()
        average_annuity_ratio = df_filtered["ANNUITY_TO_INCOME"].mean()
        high_credit_ratio = (df_filtered["CREDIT_TO_INCOME"] > 3).mean() * 100
        high_annuity_ratio = (df_filtered["ANNUITY_TO_INCOME"] > 0.3).mean() * 100

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("Average Credit / Income", f"{average_credit_ratio:.2f}x")
        metric_2.metric("Average Annuity / Income", f"{average_annuity_ratio:.2%}")
        metric_3.metric("Credit / Income Above 3x", f"{high_credit_ratio:.2f}%")
        metric_4.metric("Annuity / Income Above 30%", f"{high_annuity_ratio:.2f}%")

        left, right = st.columns(2)
        with left:
            credit_ratio_fig = px.histogram(
                df_filtered,
                x="CREDIT_TO_INCOME",
                nbins=40,
                title="Credit-to-Income Ratio",
                labels={"CREDIT_TO_INCOME": "Credit / Income"},
                color_discrete_sequence=["#1f77b4"],
            )
            credit_ratio_fig.update_layout(yaxis_title="Number of People")
            st.plotly_chart(credit_ratio_fig, use_container_width=True)

            annuity_ratio_fig = px.histogram(
                df_filtered,
                x="ANNUITY_TO_INCOME",
                nbins=40,
                title="Annuity-to-Income Ratio",
                labels={"ANNUITY_TO_INCOME": "Annuity / Income"},
                color_discrete_sequence=["#ff7f0e"],
            )
            annuity_ratio_fig.update_layout(yaxis_title="Number of People")
            st.plotly_chart(annuity_ratio_fig, use_container_width=True)

        with right:
            affordability_scatter = px.scatter(
                df_filtered,
                x="AMT_INCOME_TOTAL",
                y="AMT_CREDIT",
                color="TARGET",
                title="Income vs Credit Amount",
                labels={
                    "AMT_INCOME_TOTAL": "Annual Income",
                    "AMT_CREDIT": "Credit Amount",
                    "TARGET": "Default Status",
                },
                color_discrete_map={0: "#2ca02c", 1: "#d62728"},
                opacity=0.55,
            )
            st.plotly_chart(affordability_scatter, use_container_width=True)

            goods_ratio_fig = px.histogram(
                df_filtered,
                x="CREDIT_TO_GOODS_PRICE",
                nbins=40,
                title="Credit-to-Goods-Price Ratio",
                labels={"CREDIT_TO_GOODS_PRICE": "Credit / Goods Price"},
                color_discrete_sequence=["#9467bd"],
            )
            goods_ratio_fig.update_layout(yaxis_title="Number of People")
            st.plotly_chart(goods_ratio_fig, use_container_width=True)

        st.subheader("Default Rate by Affordability Level")
        affordability = df_filtered.copy()
        affordability["Affordability Band"] = pd.cut(
            affordability["CREDIT_TO_INCOME"],
            bins=[0, 1, 2, 3, 5, float("inf")],
            labels=["Up to 1x", "1x-2x", "2x-3x", "3x-5x", "Above 5x"],
            right=False,
        )
        affordability_summary = (
            affordability.groupby("Affordability Band", observed=True)["TARGET"]
            .agg(Applicants="count", Default_Rate="mean")
            .reset_index()
        )
        affordability_summary["Observed Default Rate %"] = affordability_summary["Default_Rate"] * 100

        affordability_fig = px.bar(
            affordability_summary,
            x="Affordability Band",
            y="Observed Default Rate %",
            title="Observed Default Rate by Credit-to-Income Band",
            color="Observed Default Rate %",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(affordability_fig, use_container_width=True)
        st.dataframe(affordability_summary[["Affordability Band", "Applicants", "Observed Default Rate %"]])

except FileNotFoundError:
    st.error("Dataset file not found. Add `data/application_train.csv`.")
