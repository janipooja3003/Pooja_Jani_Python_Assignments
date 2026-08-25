from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.filters import sidebar_filters, apply_filters

st.set_page_config(page_title="Installment Payment Analysis", layout="wide")
st.title("💳 Installment Payment Analysis")

base_path = Path(__file__).resolve().parents[1] / "data"
application_path = base_path / "application_train.csv"
installment_path = base_path / "installments_payments.csv"

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
    installments = pd.read_csv(installment_path)

    filters = sidebar_filters(applications)
    filtered_applications = apply_filters(applications, filters)

    if filtered_applications.empty:
        st.warning("No applicant records match the selected filters.")
    else:
        installments_filtered = installments[
            installments["SK_ID_CURR"].isin(filtered_applications["SK_ID_CURR"])
        ].copy()

        if installments_filtered.empty:
            st.warning("No installment records match the selected filters.")
        else:
            installments_filtered["AMT_INSTALMENT"] = pd.to_numeric(
                installments_filtered["AMT_INSTALMENT"], errors="coerce"
            )
            installments_filtered["AMT_PAYMENT"] = pd.to_numeric(
                installments_filtered["AMT_PAYMENT"], errors="coerce"
            )
            installments_filtered["DAYS_INSTALMENT"] = pd.to_numeric(
                installments_filtered["DAYS_INSTALMENT"], errors="coerce"
            )
            installments_filtered["DAYS_ENTRY_PAYMENT"] = pd.to_numeric(
                installments_filtered["DAYS_ENTRY_PAYMENT"], errors="coerce"
            )
            installments_filtered["PAYMENT_DELAY_DAYS"] = (
                installments_filtered["DAYS_ENTRY_PAYMENT"]
                - installments_filtered["DAYS_INSTALMENT"]
            )
            installments_filtered["PAYMENT_RATIO"] = (
                installments_filtered["AMT_PAYMENT"] / installments_filtered["AMT_INSTALMENT"]
            ).replace([float("inf"), -float("inf")], pd.NA)
            installments_filtered["LATE_PAYMENT"] = (
                installments_filtered["PAYMENT_DELAY_DAYS"] > 0
            )
            installments_filtered["ON_TIME_PAYMENT"] = (
                installments_filtered["PAYMENT_DELAY_DAYS"] <= 0
            )

            total_payments = len(installments_filtered)
            unique_customers = installments_filtered["SK_ID_CURR"].nunique()
            on_time_rate = (
                installments_filtered["ON_TIME_PAYMENT"].mean() * 100
                if not installments_filtered.empty
                else 0
            )
            avg_delay = installments_filtered["PAYMENT_DELAY_DAYS"].mean()
            avg_payment_ratio = installments_filtered["PAYMENT_RATIO"].mean()

            metric_1, metric_2, metric_3, metric_4 = st.columns(4)
            metric_1.metric("Installment Records", f"{total_payments:,}")
            metric_2.metric("Unique Customers", f"{unique_customers:,}")
            metric_3.metric("On-time Rate", f"{on_time_rate:.1f}%")
            metric_4.metric("Avg Delay (days)", f"{avg_delay:.1f}")

            st.subheader("Installment payment behavior")
            left, right = st.columns(2)

            with left:
                delay_hist = px.histogram(
                    installments_filtered,
                    x="PAYMENT_DELAY_DAYS",
                    nbins=40,
                    title="Payment delay distribution",
                    labels={"PAYMENT_DELAY_DAYS": "Days late (positive means late)"},
                )
                st.plotly_chart(delay_hist, use_container_width=True)

                ratio_hist = px.histogram(
                    installments_filtered.dropna(subset=["PAYMENT_RATIO"]),
                    x="PAYMENT_RATIO",
                    nbins=40,
                    title="Payment-to-instalment ratio distribution",
                    labels={"PAYMENT_RATIO": "Amount paid / amount due"},
                )
                ratio_hist.update_xaxes(range=[0, 1.5])
                st.plotly_chart(ratio_hist, use_container_width=True)

            with right:
                status_summary = (
                    installments_filtered.groupby("NUM_INSTALMENT_NUMBER", as_index=False)
                    .agg(
                        avg_delay=("PAYMENT_DELAY_DAYS", "mean"),
                        avg_payment_ratio=("PAYMENT_RATIO", "mean"),
                        payments=("SK_ID_PREV", "count"),
                    )
                    .sort_values("NUM_INSTALMENT_NUMBER")
                )
                timing_long = status_summary.melt(
                    id_vars=["NUM_INSTALMENT_NUMBER"],
                    value_vars=["avg_delay", "avg_payment_ratio"],
                    var_name="Metric",
                    value_name="Value",
                )
                timing_fig = px.line(
                    timing_long,
                    x="NUM_INSTALMENT_NUMBER",
                    y="Value",
                    color="Metric",
                    markers=True,
                    title="Installment timing and payment ratio by installment number",
                    labels={
                        "NUM_INSTALMENT_NUMBER": "Installment number",
                        "Value": "Value",
                        "Metric": "Metric",
                    },
                )
                timing_fig.update_layout(legend_title_text="Metric")
                st.plotly_chart(timing_fig, use_container_width=True)

                late_breakdown = (
                    installments_filtered["LATE_PAYMENT"]
                    .value_counts()
                    .rename_axis("Payment Status")
                    .reset_index(name="Installments")
                )
                late_breakdown["Payment Status"] = late_breakdown["Payment Status"].map(
                    {True: "Late", False: "On time or early"}
                )
                late_fig = px.pie(
                    late_breakdown,
                    names="Payment Status",
                    values="Installments",
                    title="Installment payment punctuality",
                )
                st.plotly_chart(late_fig, use_container_width=True)

            st.subheader("Per-customer drill-down")
            customer_summary = (
                installments_filtered.groupby("SK_ID_CURR", as_index=False)
                .agg(
                    total_installments=("SK_ID_PREV", "count"),
                    avg_delay=("PAYMENT_DELAY_DAYS", "mean"),
                    max_delay=("PAYMENT_DELAY_DAYS", "max"),
                    avg_payment_ratio=("PAYMENT_RATIO", "mean"),
                    late_count=("LATE_PAYMENT", "sum"),
                    total_paid=("AMT_PAYMENT", "sum"),
                    total_due=("AMT_INSTALMENT", "sum"),
                )
                .merge(
                    filtered_applications[["SK_ID_CURR", "TARGET"]].drop_duplicates(),
                    on="SK_ID_CURR",
                    how="left",
                )
            )
            customer_summary["Default Flag"] = customer_summary["TARGET"].map(
                {0: "No default", 1: "Default"}
            )

            scatter_fig = px.scatter(
                customer_summary,
                x="avg_delay",
                y="avg_payment_ratio",
                size="total_installments",
                color="Default Flag",
                hover_name="SK_ID_CURR",
                title="Applicant drill-down: average delay vs payment ratio",
                labels={
                    "avg_delay": "Average payment delay (days)",
                    "avg_payment_ratio": "Average payment / instalment",
                    "total_installments": "Installments",
                },
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

            selected_customer = st.selectbox(
                "Select an applicant",
                sorted(installments_filtered["SK_ID_CURR"].unique().tolist()),
            )
            selected_rows = installments_filtered[
                installments_filtered["SK_ID_CURR"] == selected_customer
            ].sort_values("NUM_INSTALMENT_NUMBER").copy()

            selected_rows["Amount Paid %"] = (
                selected_rows["AMT_PAYMENT"] / selected_rows["AMT_INSTALMENT"]
            )
            selected_detail = px.line(
                selected_rows,
                x="NUM_INSTALMENT_NUMBER",
                y=["AMT_INSTALMENT", "AMT_PAYMENT", "PAYMENT_DELAY_DAYS"],
                markers=True,
                title=f"Selected applicant {selected_customer}: installment amount and delay tracking",
                labels={
                    "NUM_INSTALMENT_NUMBER": "Installment number",
                    "value": "Value",
                },
            )
            selected_detail.update_layout(legend_title_text="Measure")
            st.plotly_chart(selected_detail, use_container_width=True)

            st.dataframe(
                selected_rows[
                    [
                        "SK_ID_PREV",
                        "NUM_INSTALMENT_NUMBER",
                        "DAYS_INSTALMENT",
                        "DAYS_ENTRY_PAYMENT",
                        "PAYMENT_DELAY_DAYS",
                        "AMT_INSTALMENT",
                        "AMT_PAYMENT",
                        "PAYMENT_RATIO",
                    ]
                ].reset_index(drop=True),
                use_container_width=True,
            )

except FileNotFoundError:
    st.error("Required dataset not found. Add application_train.csv and installments_payments.csv to the data/ folder.")
