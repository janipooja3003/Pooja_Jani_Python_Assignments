import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import load_application_train_data
from utils.filters import sidebar_filters, apply_filters

st.set_page_config(page_title="Income Analysis")
st.title("💰 Income Analysis")

try:
    df = load_application_train_data()
    filters = sidebar_filters(df)
    df_filtered = apply_filters(df, filters)

    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
    else:
        df_filtered = df_filtered.copy()

        income_cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]

        st.subheader("Income Summary")
        income_summary = df_filtered[income_cols].describe().T
        income_summary = income_summary.rename(index={
            "AMT_INCOME_TOTAL": "Annual Income",
            "AMT_CREDIT": "Credit Amount",
            "AMT_ANNUITY": "Annuity",
            "AMT_GOODS_PRICE": "Goods Price",
        })
        st.dataframe(income_summary)

        col1, col2 = st.columns(2)

        with col1:
            income_hist = px.histogram(
                df_filtered,
                x="AMT_INCOME_TOTAL",
                nbins=40,
                title="Annual Income Distribution",
                labels={"AMT_INCOME_TOTAL": "Annual Income"},
                color_discrete_sequence=["#1f77b4"],
            )
            income_hist.update_layout(xaxis_title="Annual Income", yaxis_title="Number of People")
            st.plotly_chart(income_hist, use_container_width=True)

            income_box = px.box(
                df_filtered,
                y="AMT_INCOME_TOTAL",
                title="Annual Income Outlier Distribution",
                labels={"AMT_INCOME_TOTAL": "Annual Income"},
            )
            income_box.update_layout(yaxis_title="Annual Income")
            st.plotly_chart(income_box, use_container_width=True)

        with col2:
            credit_hist = px.histogram(
                df_filtered,
                x="AMT_CREDIT",
                nbins=40,
                title="Credit Amount Distribution",
                labels={"AMT_CREDIT": "Credit Amount"},
                color_discrete_sequence=["#ff7f0e"],
            )
            credit_hist.update_layout(xaxis_title="Credit Amount", yaxis_title="Number of People")
            st.plotly_chart(credit_hist, use_container_width=True)

            income_income_type = (
                df_filtered.groupby("NAME_INCOME_TYPE", dropna=False)["AMT_INCOME_TOTAL"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            income_income_type.columns = ["Income Type", "Average Annual Income"]
            income_type_fig = px.bar(
                income_income_type,
                x="Income Type",
                y="Average Annual Income",
                title="Average Annual Income by Income Type",
                color="Income Type",
            )
            st.plotly_chart(income_type_fig, use_container_width=True)

        st.subheader("Income by Education Type")
        edu_income = (
            df_filtered.groupby("NAME_EDUCATION_TYPE", dropna=False)["AMT_INCOME_TOTAL"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        edu_income.columns = ["Education Type", "Average Annual Income"]
        edu_income_fig = px.bar(
            edu_income,
            x="Education Type",
            y="Average Annual Income",
            title="Average Annual Income by Education",
            color="Education Type",
        )
        st.plotly_chart(edu_income_fig, use_container_width=True)

        st.subheader("Income vs Credit")
        scatter_fig = px.scatter(
            df_filtered,
            x="AMT_INCOME_TOTAL",
            y="AMT_CREDIT",
            title="Income vs Credit Amount",
            labels={"AMT_INCOME_TOTAL": "Annual Income", "AMT_CREDIT": "Credit Amount"},
            opacity=0.6,
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

        st.subheader("Default Rate by Income Band")
        income_band_df = df_filtered.copy()
        income_band_df["Income Band"] = pd.cut(
            income_band_df["AMT_INCOME_TOTAL"],
            bins=[0, 25000, 50000, 100000, 150000, 200000, float("inf")],
            labels=["Below 25K", "25K-50K", "50K-100K", "100K-150K", "150K-200K", "200K+"],
            right=False,
        )

        band_summary = (
            income_band_df.groupby("Income Band", dropna=False)["TARGET"]
            .agg(["count", "mean"])
            .reset_index()
        )
        band_summary["Default Rate %"] = band_summary["mean"] * 100
        band_summary = band_summary.rename(columns={"count": "Applicants", "mean": "Observed Default Rate"})

        band_fig = px.bar(
            band_summary,
            x="Income Band",
            y="Default Rate %",
            title="Observed Default Rate by Income Band",
            color="Default Rate %",
            color_continuous_scale="RdYlGn_r",
        )
        st.plotly_chart(band_fig, use_container_width=True)

        st.dataframe(band_summary[["Income Band", "Applicants", "Observed Default Rate", "Default Rate %"]])

        lowest_income_band = band_summary.loc[band_summary["Default Rate %"].idxmin(), "Income Band"] if not band_summary.empty else None
        highest_income_band = band_summary.loc[band_summary["Default Rate %"].idxmax(), "Income Band"] if not band_summary.empty else None

        # if not band_summary.empty:
        #     st.markdown(
        #         "### Answer: Do lower-income applicants have a higher observed default rate?"
        #     )
        #     st.write(
        #         f"Based on the current filtered dataset, the highest observed default rate appears in the {highest_income_band} band, while the lowest is in the {lowest_income_band} band."
        #     )
        #     st.write(
        #         "This suggests the relationship is not uniform across all bands, so the default pattern should be interpreted by income segment rather than by income alone."
        #     )

except FileNotFoundError:
    st.error("Dataset file not found. Add `data/application_train.csv`.")
