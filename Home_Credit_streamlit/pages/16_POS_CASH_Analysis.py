# Point of Sale Cash loans
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.filters import sidebar_filters, apply_filters

st.set_page_config(page_title="POS Cash Analysis", layout="wide")
st.title("📋 POS Cash Analysis")

base_path = Path(__file__).resolve().parents[1] / "data"
application_path = base_path / "application_train.csv"
pos_path = base_path / "POS_CASH_balance.csv"

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
    pos = pd.read_csv(pos_path)

    filters = sidebar_filters(applications)
    filtered_applications = apply_filters(applications, filters)

    if filtered_applications.empty:
        st.warning("No applicant records match the selected filters.")
    else:
        pos_filtered = pos[pos["SK_ID_CURR"].isin(filtered_applications["SK_ID_CURR"])].copy()

        if pos_filtered.empty:
            st.warning("No POS cash records match the selected filters.")
        else:
            pos_filtered["CNT_INSTALMENT"] = pd.to_numeric(
                pos_filtered["CNT_INSTALMENT"], errors="coerce"
            )
            pos_filtered["CNT_INSTALMENT_FUTURE"] = pd.to_numeric(
                pos_filtered["CNT_INSTALMENT_FUTURE"], errors="coerce"
            )
            pos_filtered["SK_DPD"] = pd.to_numeric(pos_filtered["SK_DPD"], errors="coerce")
            pos_filtered["SK_DPD_DEF"] = pd.to_numeric(
                pos_filtered["SK_DPD_DEF"], errors="coerce"
            )

            total_records = len(pos_filtered)
            unique_customers = pos_filtered["SK_ID_CURR"].nunique()
            active_contracts = (pos_filtered["NAME_CONTRACT_STATUS"] == "Active").sum()
            avg_dpd = float(pos_filtered["SK_DPD"].mean())
            avg_future_instalments = float(pos_filtered["CNT_INSTALMENT_FUTURE"].mean())

            metric_1, metric_2, metric_3, metric_4 = st.columns(4)
            metric_1.metric("POS Records", f"{total_records:,}")
            metric_2.metric("Unique Customers", f"{unique_customers:,}")
            metric_3.metric("Active Contracts", f"{active_contracts:,}")
            metric_4.metric("Avg DPD", f"{avg_dpd:.2f}")

            st.subheader("Exploratory POS/CASH visuals")

            left_col, right_col = st.columns(2)

            with left_col:
                status_counts = (
                    pos_filtered["NAME_CONTRACT_STATUS"]
                    .fillna("Unknown")
                    .value_counts()
                    .rename_axis("Contract Status")
                    .reset_index(name="Records")
                )
                status_fig = px.bar(
                    status_counts,
                    x="Contract Status",
                    y="Records",
                    color="Contract Status",
                    title="Contract status breakdown",
                    labels={"Records": "Number of records"},
                )
                st.plotly_chart(status_fig, use_container_width=True)

                progress_df = pos_filtered.dropna(
                    subset=["CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE"]
                ).copy()
                progress_df["Instalment Progress"] = (
                    (progress_df["CNT_INSTALMENT"] - progress_df["CNT_INSTALMENT_FUTURE"])
                    / progress_df["CNT_INSTALMENT"]
                ).clip(lower=0, upper=1)
                progress_summary = (
                    progress_df.groupby("NAME_CONTRACT_STATUS", as_index=False)[
                        "Instalment Progress"
                    ]
                    .mean()
                    .sort_values("Instalment Progress")
                )
                progress_fig = px.bar(
                    progress_summary,
                    x="NAME_CONTRACT_STATUS",
                    y="Instalment Progress",
                    color="NAME_CONTRACT_STATUS",
                    title="Average installment progress by contract status",
                    labels={"NAME_CONTRACT_STATUS": "Contract Status"},
                )
                progress_fig.update_yaxes(range=[0, 1])
                st.plotly_chart(progress_fig, use_container_width=True)

            with right_col:
                dpd_trend = (
                    pos_filtered.groupby("MONTHS_BALANCE", as_index=False)
                    .agg(avg_dpd=("SK_DPD", "mean"), max_dpd=("SK_DPD", "max"))
                    .sort_values("MONTHS_BALANCE")
                )
                dpd_fig = px.line(
                    dpd_trend,
                    x="MONTHS_BALANCE",
                    y=["avg_dpd", "max_dpd"],
                    markers=True,
                    title="DPD trend over month balance",
                    labels={"value": "DPD", "MONTHS_BALANCE": "Months balance"},
                )
                dpd_fig.update_layout(legend_title_text="Metric")
                st.plotly_chart(dpd_fig, use_container_width=True)

                progress_hist = px.histogram(
                    progress_df,
                    x="Instalment Progress",
                    nbins=30,
                    color="NAME_CONTRACT_STATUS",
                    barmode="overlay",
                    opacity=0.6,
                    title="Instalment progress distribution",
                    labels={"Instalment Progress": "Completed share"},
                )
                progress_hist.update_xaxes(range=[0, 1])
                st.plotly_chart(progress_hist, use_container_width=True)

            st.subheader("Per-applicant drill-down")
            customer_profile = (
                pos_filtered.groupby("SK_ID_CURR", as_index=False)
                .agg(
                    records=("SK_ID_PREV", "count"),
                    latest_month=("MONTHS_BALANCE", "max"),
                    avg_dpd=("SK_DPD", "mean"),
                    max_dpd=("SK_DPD", "max"),
                    avg_future_instalments=("CNT_INSTALMENT_FUTURE", "mean"),
                    avg_instalment=("CNT_INSTALMENT", "mean"),
                )
                .merge(
                    filtered_applications[["SK_ID_CURR", "TARGET"]].drop_duplicates(),
                    on="SK_ID_CURR",
                    how="left",
                )
            )
            customer_profile["Default Flag"] = customer_profile["TARGET"].map(
                {0: "No default", 1: "Default"}
            )

            applicant_fig = px.scatter(
                customer_profile,
                x="avg_dpd",
                y="avg_future_instalments",
                size="records",
                color="Default Flag",
                hover_name="SK_ID_CURR",
                title="Applicant drill-down: DPD vs future installments",
                labels={
                    "avg_dpd": "Average DPD",
                    "avg_future_instalments": "Average future installments",
                    "records": "POS records",
                },
            )
            st.plotly_chart(applicant_fig, use_container_width=True)

            selected_customer = st.selectbox(
                "Select applicant for detailed tracking",
                sorted(pos_filtered["SK_ID_CURR"].unique().tolist()),
            )
            selected_rows = (
                pos_filtered[pos_filtered["SK_ID_CURR"] == selected_customer]
                .sort_values("MONTHS_BALANCE")
                .copy()
            )
            selected_rows["Instalment Progress"] = (
                (selected_rows["CNT_INSTALMENT"] - selected_rows["CNT_INSTALMENT_FUTURE"])
                / selected_rows["CNT_INSTALMENT"]
            ).clip(lower=0, upper=1)

            detail_fig = px.line(
                selected_rows,
                x="MONTHS_BALANCE",
                y=["CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "SK_DPD"],
                markers=True,
                title=f"Selected applicant {selected_customer}: instalments and DPD history",
                labels={"value": "Value", "MONTHS_BALANCE": "Month balance"},
            )
            detail_fig.update_layout(legend_title_text="Measure")
            st.plotly_chart(detail_fig, use_container_width=True)

            st.dataframe(
                selected_rows[
                    [
                        "SK_ID_PREV",
                        "MONTHS_BALANCE",
                        "NAME_CONTRACT_STATUS",
                        "CNT_INSTALMENT",
                        "CNT_INSTALMENT_FUTURE",
                        "SK_DPD",
                        "SK_DPD_DEF",
                        "Instalment Progress",
                    ]
                ].reset_index(drop=True),
                use_container_width=True,
            )

except FileNotFoundError:
    st.error("Required dataset not found. Add application_train.csv and POS_CASH_balance.csv to the data/ folder.")

