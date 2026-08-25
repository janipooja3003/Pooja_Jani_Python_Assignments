#An outlier is a value that is unusually far from the rest of the data.
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.feature_engineering import calc_kpis
from utils.preprocessing import load_application_train_data
from utils.filters import sidebar_filters, apply_filters
from utils.feature_engineering import calc_kpis #, top_bottom_summary
from utils.charts import line_chart, bar_chart, scatter_chart


st.set_page_config(page_title="Outlier Analysis")

st.title("🔍 Outlier Analysis")

try:
    df = load_application_train_data()
    filters = sidebar_filters(df)
    df_filtered = apply_filters(df, filters)

    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
    else:
        numerical_cols = ["AMT_INCOME_TOTAL", #non-missing values in the AMT_INCOME_TOTAL column
                          "AMT_CREDIT",
                          "AMT_ANNUITY",
                          "AMT_GOODS_PRICE",
                          "DAYS_BIRTH",
                          "DAYS_EMPLOYED",
                          "CNT_CHILDREN",
                          "CNT_FAM_MEMBERS",
                          "AMT_REQ_CREDIT_BUREAU_YEAR"]
        describe_df = df_filtered[numerical_cols].describe().T
        describe_df["99.5%"] = df_filtered[numerical_cols].quantile(0.995)
        st.subheader("Numerical Columns Summary")
        st.dataframe(describe_df)

        metric = st.selectbox("Select metric for bell curve analysis", numerical_cols)
        series = df_filtered[metric].dropna()
        p995 = float(series.quantile(0.995))
        series = series[series <= p995].copy()

        if not series.empty:
            mean_value = float(series.mean())
            median_value = float(series.median())
            mode_value = float(series.mode().iloc[0]) if not series.mode().empty else None
            max_value = float(series.max())

            st.subheader(f"Bell Curve: {metric} (values <= 99.5th percentile)")
            stats_cols = st.columns(4)
            stats_cols[0].metric("Mean", f"{mean_value:,.2f}")
            stats_cols[1].metric("Median", f"{median_value:,.2f}")
            stats_cols[2].metric("Mode", f"{mode_value:,.2f}" if mode_value is not None else "N/A")
            stats_cols[3].metric("Max", f"{max_value:,.2f}")

            hist_fig = px.histogram(
                x=series,
                nbins=40,
                title=f"Frequency Distribution of {metric}",
                labels={"x": metric, "y": "Number of People"},
                opacity=0.7,
            )

            std_value = float(series.std(ddof=0))
            if std_value == 0:
                std_value = 1.0
            x_curve = np.linspace(series.min(), series.max(), 400)
            y_curve = (1 / (std_value * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_curve - mean_value) / std_value) ** 2)
            scale = max(series.value_counts().max(), 1)
            hist_fig.add_trace(
                go.Scatter(
                    x=x_curve,
                    y=y_curve * scale * 0.9,
                    mode="lines",
                    name="Bell Curve",
                    line=dict(color="red", width=3),
                )
            )

            for label, value in [("Mean", mean_value), ("Median", median_value), ("Mode", mode_value), ("Max", max_value)]:
                if value is None:
                    continue
                hist_fig.add_vline(
                    x=value,
                    line_dash="dash",
                    line_color="navy" if label == "Mean" else "darkgreen" if label == "Median" else "orange" if label == "Mode" else "purple",
                    annotation_text=label,
                    annotation_position="top left",
                )

            hist_fig.update_layout(
                xaxis_title=metric,
                yaxis_title="Number of People",
                legend_title_text="Metrics",
                template="plotly_white",
            )
            st.plotly_chart(hist_fig, use_container_width=True)

            no_max_series = series[series < max_value].copy()
            if not no_max_series.empty:
                no_max_max = float(no_max_series.max())
                st.subheader(f"Max Value Without Outlier: {no_max_max:,.2f}")
                st.metric("Max Value", f"{no_max_max:,.2f}")
                no_max_mean = float(no_max_series.mean())
                no_max_median = float(no_max_series.median())
                no_max_mode = float(no_max_series.mode().iloc[0]) if not no_max_series.mode().empty else None

                st.subheader(f"Bell Curve Without Max Value: {metric}")
                st.subheader(f"Bell Curve: {metric}")
                stats_cols = st.columns(4)
                stats_cols[0].metric("Mean", f"{no_max_mean:,.2f}")
                stats_cols[1].metric("Median", f"{no_max_median:,.2f}")
                stats_cols[2].metric("Mode", f"{no_max_mode:,.2f}" if no_max_mode is not None else "N/A")
                stats_cols[3].metric("Max", f"{no_max_max:,.2f}")
                no_max_fig = px.histogram(
                    x=no_max_series,
                    nbins=40,
                    title=f"Frequency Distribution of {metric} (Without Max Value)",
                    labels={"x": metric, "y": "Number of People"},
                    opacity=0.7,
                )

                no_max_std = float(no_max_series.std(ddof=0))
                if no_max_std == 0:
                    no_max_std = 1.0
                no_max_x = np.linspace(no_max_series.min(), no_max_series.max(), 400)
                no_max_y = (1 / (no_max_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((no_max_x - no_max_mean) / no_max_std) ** 2)
                no_max_scale = max(no_max_series.value_counts().max(), 1)
                no_max_fig.add_trace(
                    go.Scatter(
                        x=no_max_x,
                        y=no_max_y * no_max_scale * 0.9,
                        mode="lines",
                        name="Bell Curve",
                        line=dict(color="red", width=3),
                    )
                )

                for label, value in [("Mean", no_max_mean), ("Median", no_max_median), ("Mode", no_max_mode)]:
                    if value is None:
                        continue
                    no_max_fig.add_vline(
                        x=value,
                        line_dash="dash",
                        line_color="navy" if label == "Mean" else "darkgreen" if label == "Median" else "orange",
                        annotation_text=label,
                        annotation_position="top left",
                    )

                no_max_fig.update_layout(
                    xaxis_title=metric,
                    yaxis_title="Number of People",
                    legend_title_text="Metrics",
                    template="plotly_white",
                )
                st.plotly_chart(no_max_fig, use_container_width=True)

                st.subheader(f"Scatter Plot Without Max Value: {metric}")
                scatter_df = df_filtered[[metric]].copy().reset_index(drop=True)
                scatter_df["row_id"] = np.arange(len(scatter_df))
                scatter_df = scatter_df[scatter_df[metric] < max_value].copy()
                scatter_fig = px.scatter(
                    scatter_df,
                    x=metric,
                    y="row_id",
                    title=f"{metric} Distribution Without Highest Value",
                    labels={metric: metric, "row_id": "Number of People"},
                    opacity=0.6,
                    color_discrete_sequence=["#2E8B57"],
                )
                scatter_fig.add_hline(
                    y=scatter_df["row_id"].median(),
                    line_dash="dash",
                    line_color="darkgreen",
                    annotation_text="Median",
                )
                scatter_fig.add_hline(
                    y=scatter_df["row_id"].mean(),
                    line_dash="dot",
                    line_color="navy",
                    annotation_text="Mean",
                )
                scatter_fig.update_layout(
                    xaxis_title=metric,
                    yaxis_title="Number of People",
                    template="plotly_white",
                )
                st.plotly_chart(scatter_fig, use_container_width=True)

    fig = px.box(
                df_filtered,
                y="AMT_INCOME_TOTAL",
                title="Income Outlier Analysis")
    st.plotly_chart(
                fig,
                use_container_width=True )
                                                    
except FileNotFoundError:
    st.error("Dataset file not found. Add `data/application_train.csv`.")
