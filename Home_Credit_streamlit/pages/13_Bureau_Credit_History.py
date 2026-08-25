from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Bureau Credit History")
st.title("🏦 Bureau Credit History Analysis")

bureau_path = Path(__file__).resolve().parents[1] / "data" / "bureau.csv"

try:
    bureau = pd.read_csv(bureau_path)

    with st.sidebar:
        st.header("Bureau Filters")
        active_options = sorted(bureau["CREDIT_ACTIVE"].dropna().unique())
        selected_active = st.multiselect(
            "Credit Status",
            active_options,
            default=active_options,
        )
        type_options = sorted(bureau["CREDIT_TYPE"].dropna().unique())
        selected_types = st.multiselect(
            "Credit Type",
            type_options,
            default=type_options,
        )

    filtered = bureau[
        bureau["CREDIT_ACTIVE"].isin(selected_active)
        & bureau["CREDIT_TYPE"].isin(selected_types)
    ].copy()

    if filtered.empty:
        st.warning("No bureau records match the selected filters.")
    else:
        filtered["CREDIT_AGE_YEARS"] = filtered["DAYS_CREDIT"].abs() / 365.25
        filtered["OVERDUE_STATUS"] = filtered["CREDIT_DAY_OVERDUE"].gt(0).map(
            {True: "Overdue", False: "Not Overdue"}
        )

        total_records = len(filtered)
        customers = filtered["SK_ID_CURR"].nunique()
        overdue_records = int(filtered["OVERDUE_STATUS"].eq("Overdue").sum())
        overdue_rate = overdue_records / total_records * 100
        total_debt = filtered["AMT_CREDIT_SUM_DEBT"].sum()

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("Credit Records", f"{total_records:,}")
        metric_2.metric("Customers", f"{customers:,}")
        metric_3.metric("Overdue Record Rate", f"{overdue_rate:.2f}%")
        metric_4.metric("Total Reported Debt", f"${total_debt:,.0f}")

        st.subheader("Credit Status and History")
        left, right = st.columns(2)

        with left:
            active_data = (
                filtered["CREDIT_ACTIVE"]
                .value_counts()
                .rename_axis("Credit Status")
                .reset_index(name="Records")
            )
            active_fig = px.bar(
                active_data,
                x="Credit Status",
                y="Records",
                color="Credit Status",
                title="Credit Records by Status",
            )
            st.plotly_chart(active_fig, use_container_width=True)

            age_fig = px.histogram(
                filtered,
                x="CREDIT_AGE_YEARS",
                nbins=40,
                title="Credit History Age Distribution",
                labels={"CREDIT_AGE_YEARS": "Years Since Credit Was Reported"},
                color_discrete_sequence=["#1f77b4"],
            )
            age_fig.update_layout(yaxis_title="Number of Credit Records")
            st.plotly_chart(age_fig, use_container_width=True)

        with right:
            overdue_data = (
                filtered["OVERDUE_STATUS"]
                .value_counts()
                .rename_axis("Overdue Status")
                .reset_index(name="Records")
            )
            overdue_fig = px.bar(
                overdue_data,
                x="Overdue Status",
                y="Records",
                color="Overdue Status",
                color_discrete_map={"Overdue": "#d62728", "Not Overdue": "#2ca02c"},
                title="Overdue vs Not Overdue Records",
            )
            st.plotly_chart(overdue_fig, use_container_width=True)

            overdue_days_fig = px.histogram(
                filtered[filtered["CREDIT_DAY_OVERDUE"] > 0],
                x="CREDIT_DAY_OVERDUE",
                nbins=40,
                title="Overdue Days Distribution",
                labels={"CREDIT_DAY_OVERDUE": "Days Overdue"},
                color_discrete_sequence=["#d62728"],
            )
            overdue_days_fig.update_layout(yaxis_title="Number of Credit Records")
            st.plotly_chart(overdue_days_fig, use_container_width=True)

        st.subheader("Credit Type Analysis")
        type_summary = (
            filtered.groupby("CREDIT_TYPE")
            .agg(
                Records=("SK_ID_BUREAU", "count"),
                Customers=("SK_ID_CURR", "nunique"),
                Average_Credit=("AMT_CREDIT_SUM", "mean"),
                Average_Debt=("AMT_CREDIT_SUM_DEBT", "mean"),
                Overdue_Rate=("OVERDUE_STATUS", lambda values: (values == "Overdue").mean() * 100),
            )
            .reset_index()
            .sort_values("Records", ascending=False)
        )
        type_summary = type_summary.rename(
            columns={
                "CREDIT_TYPE": "Credit Type",
                "Average_Credit": "Average Credit",
                "Average_Debt": "Average Debt",
                "Overdue_Rate": "Overdue Rate %",
            }
        )
        type_fig = px.bar(
            type_summary.head(15),
            x="Credit Type",
            y="Records",
            color="Overdue Rate %",
            color_continuous_scale="Reds",
            title="Top Credit Types by Record Count",
            hover_data=["Customers", "Average Credit", "Average Debt", "Overdue Rate %"],
        )
        st.plotly_chart(type_fig, use_container_width=True)
        st.dataframe(type_summary)

        st.subheader("Debt and Credit Exposure")
        exposure_data = filtered[
            ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE"]
        ].describe().T
        exposure_data.index = ["Credit Sum", "Debt", "Credit Limit", "Overdue Amount"]
        st.dataframe(exposure_data)

        exposure_fig = px.scatter(
            filtered.sample(min(10000, len(filtered)), random_state=42),
            x="AMT_CREDIT_SUM",
            y="AMT_CREDIT_SUM_DEBT",
            color="OVERDUE_STATUS",
            title="Credit Amount vs Reported Debt",
            labels={
                "AMT_CREDIT_SUM": "Credit Amount",
                "AMT_CREDIT_SUM_DEBT": "Reported Debt",
            },
            color_discrete_map={"Overdue": "#d62728", "Not Overdue": "#2ca02c"},
            opacity=0.55,
        )
        st.plotly_chart(exposure_fig, use_container_width=True)

except FileNotFoundError:
    st.error("Bureau dataset not found. Add `data/bureau.csv`.")
