from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.filters import sidebar_filters, apply_filters

st.set_page_config(page_title="Previous Applications")
st.title("📋 Previous Applications Analysis")

base_path = Path(__file__).resolve().parents[1] / "data"
application_path = base_path / "application_train.csv"
previous_path = base_path / "previous_application.csv"

try:
    application_columns = [
        "SK_ID_CURR",
        "TARGET",
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
    ]
    applications = pd.read_csv(application_path, usecols=application_columns)
    previous = pd.read_csv(previous_path)

    filters = sidebar_filters(applications)
    filtered_applications = apply_filters(applications, filters)
    joined = previous.merge(
        filtered_applications[["SK_ID_CURR", "TARGET"]],
        on="SK_ID_CURR",
        how="inner",
    )

    if joined.empty:
        st.warning("No previous applications match the selected filters.")
    else:
        joined = joined.copy()
        joined["Current Outcome"] = joined["TARGET"].map(
            {0: "Non-defaulted", 1: "Defaulted"}
        )
        joined["AMT_APPLICATION"] = joined["AMT_APPLICATION"].replace(0, pd.NA)
        joined["APPROVAL_RATE"] = joined["NAME_CONTRACT_STATUS"].eq("Approved")

        records = len(joined)
        customers = joined["SK_ID_CURR"].nunique()
        approved = int(joined["APPROVAL_RATE"].sum())
        approval_rate = approved / records * 100

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("Previous Application Records", f"{records:,}")
        metric_2.metric("Customers with History", f"{customers:,}")
        metric_3.metric("Approved Previous Applications", f"{approved:,}")
        metric_4.metric("Approval Rate", f"{approval_rate:.2f}%")

        st.subheader("Previous Application Outcomes")
        outcome_data = (
            joined["NAME_CONTRACT_STATUS"]
            .value_counts()
            .rename_axis("Previous Status")
            .reset_index(name="Applications")
        )
        outcome_fig = px.bar(
            outcome_data,
            x="Previous Status",
            y="Applications",
            color="Previous Status",
            title="Previous Applications by Contract Status",
        )
        st.plotly_chart(outcome_fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            contract_data = (
                joined["NAME_CONTRACT_TYPE"]
                .value_counts()
                .rename_axis("Contract Type")
                .reset_index(name="Applications")
            )
            contract_fig = px.bar(
                contract_data,
                x="Contract Type",
                y="Applications",
                color="Contract Type",
                title="Previous Applications by Contract Type",
            )
            st.plotly_chart(contract_fig, use_container_width=True)

            amount_fig = px.histogram(
                joined,
                x="AMT_APPLICATION",
                color="NAME_CONTRACT_STATUS",
                nbins=40,
                barmode="overlay",
                opacity=0.65,
                title="Previous Requested Amount by Status",
                labels={"AMT_APPLICATION": "Requested Amount"},
            )
            amount_fig.update_layout(yaxis_title="Number of Applications")
            st.plotly_chart(amount_fig, use_container_width=True)

        with right:
            decision_fig = px.histogram(
                joined,
                x="DAYS_DECISION",
                color="NAME_CONTRACT_STATUS",
                nbins=40,
                barmode="overlay",
                opacity=0.65,
                title="Decision Timing by Status",
                labels={"DAYS_DECISION": "Days Before Current Application"},
            )
            decision_fig.update_layout(yaxis_title="Number of Applications")
            st.plotly_chart(decision_fig, use_container_width=True)

            product_data = (
                joined["NAME_PRODUCT_TYPE"]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("Product Type")
                .reset_index(name="Applications")
            )
            product_fig = px.bar(
                product_data,
                x="Product Type",
                y="Applications",
                color="Product Type",
                title="Previous Applications by Product Type",
            )
            st.plotly_chart(product_fig, use_container_width=True)

        st.subheader("Previous History and Current Default Risk")
        customer_history = (
            joined.groupby(["SK_ID_CURR", "Current Outcome"], as_index=False)
            .agg(
                Previous_Applications=("SK_ID_PREV", "count"),
                Approved_Applications=("APPROVAL_RATE", "sum"),
                Average_Previous_Credit=("AMT_CREDIT", "mean"),
                Average_Requested_Amount=("AMT_APPLICATION", "mean"),
            )
        )
        customer_history["Approval Rate %"] = (
            customer_history["Approved_Applications"]
            / customer_history["Previous_Applications"]
            * 100
        )

        history_summary = (
            customer_history.groupby("Current Outcome")
            .agg(
                Customers=("SK_ID_CURR", "count"),
                Average_Previous_Applications=("Previous_Applications", "mean"),
                Average_Approval_Rate=("Approval Rate %", "mean"),
                Average_Previous_Credit=("Average_Previous_Credit", "mean"),
            )
            .reset_index()
        )
        st.dataframe(history_summary)

        history_fig = px.box(
            customer_history,
            x="Current Outcome",
            y="Previous_Applications",
            color="Current Outcome",
            title="Previous Application Count by Current Default Outcome",
            labels={"Previous_Applications": "Number of Previous Applications"},
        )
        st.plotly_chart(history_fig, use_container_width=True)

        st.subheader("Previous Status by Current Default Outcome")
        status_risk = (
            joined.groupby(["NAME_CONTRACT_STATUS", "Current Outcome"])
            .size()
            .reset_index(name="Applications")
        )
        status_risk_fig = px.bar(
            status_risk,
            x="NAME_CONTRACT_STATUS",
            y="Applications",
            color="Current Outcome",
            barmode="group",
            title="Previous Contract Status Compared with Current Outcome",
            labels={"NAME_CONTRACT_STATUS": "Previous Contract Status"},
        )
        st.plotly_chart(status_risk_fig, use_container_width=True)

except FileNotFoundError:
    st.error("Required dataset not found. Add application_train.csv and previous_application.csv to data/.")
