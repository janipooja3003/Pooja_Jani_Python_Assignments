import streamlit as st
import pandas as pd
import plotly.express as px
from utils.feature_engineering import calc_kpis
from utils.preprocessing import load_application_train_data
from utils.filters import sidebar_filters, apply_filters
from utils.feature_engineering import calc_kpis #, top_bottom_summary
from utils.charts import line_chart, bar_chart, scatter_chart

st.set_page_config(page_title="Executive Overview")

st.title("Executive Overview")

try:
    df = load_application_train_data()
    filters = sidebar_filters(df)
    df_filtered = apply_filters(df, filters)

    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
    else:
        metrics = calc_kpis(df_filtered)
        cols = st.columns(5)
        cols[0].metric("Applications", f"{metrics['total_applications']:,}")
        cols[1].metric("Default Rate", f"{metrics['default_rate']:.2f}%")
        cols[2].metric("Avg Income", f"${metrics['avg_income']:,.0f}")
        cols[3].metric("Avg Credit", f"${metrics['avg_credit']:,.0f}")
        cols[4].metric("Max Credit", f"${metrics['max_credit']:,.0f}")

        outcome_counts = (
            df_filtered["TARGET"].value_counts().rename_axis("Status").reset_index(name="Applications")
        )
        outcome_counts["Status"] = outcome_counts["Status"].map({0: "Repayed", 1: "Defaulted"})

        fig = px.bar(
            outcome_counts,
            x="Status",
            y="Applications",
            title="Applications by Outcome",
            text="Applications",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# Credit and income analysis
# ---------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df_filtered,
            x="AMT_CREDIT",
            nbins=50,
            title="Credit Amount Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df_filtered,
            x="AMT_INCOME_TOTAL",
            nbins=50,
            title="Income Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

   #Default rate by income group

    df["INCOME_GROUP"] = pd.qcut(
    df["AMT_INCOME_TOTAL"],
    q=5,
    labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )

    income_default = (
        df.groupby("INCOME_GROUP", observed=True)["TARGET"]
        .mean()
        .mul(100)
        .reset_index(name="DEFAULT_RATE")
    )
    st.subheader("Default Rate by Income Group")

    st.bar_chart(
        income_default.set_index("INCOME_GROUP")["DEFAULT_RATE"]
    )

#Default rate by contract type
    contract_default = (
        df.groupby("NAME_CONTRACT_TYPE")["TARGET"]
        .mean()
        .mul(100)
        .reset_index(name="DEFAULT_RATE")
    )
    st.subheader("Default Rate by Contract Type")

    st.bar_chart(
        contract_default.set_index("NAME_CONTRACT_TYPE")["DEFAULT_RATE"]
    )

#Default rate by income group
    income_default = (
        df.groupby("INCOME_GROUP", observed=True)["TARGET"]
        .mean()
        .mul(100)
        .reset_index(name="DEFAULT_RATE")
    ) 
    st.subheader("Default Rate by Income Group")
    st.bar_chart(
            income_default.set_index("INCOME_GROUP")["DEFAULT_RATE"])





except FileNotFoundError:
    st.error("Dataset file not found. Add `data/application_train.csv`.")


