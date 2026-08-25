# 3. 🕳️ Missing Value Analysis
# This page should go deeper into missing data.
# KPIs
# Columns with missing values
# Total missing cells
# Highest missing %
# Columns with >50% missing
# Main chart
# px.bar(
#     # missing_df.head(20),
#     x="Missing_Percentage",
#     y="Column",
#     orientation="h"
# )
# Useful business insight
# For example:
# "Occupation type has substantial missing values and should be treated carefully before modeling."
# You can also create:
# Missing % categories
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.feature_engineering import calc_kpis
from utils.preprocessing import load_application_train_data
from utils.filters import sidebar_filters, apply_filters
from utils.feature_engineering import calc_kpis #, top_bottom_summary
from utils.charts import line_chart, bar_chart, scatter_chart


st.set_page_config(page_title="Missing Value Analysis")

st.title(" 🔍 Missing value Analysis")

try:
    df = load_application_train_data()
    filters = sidebar_filters(df)
    df_filtered = apply_filters(df, filters)

    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
    else:
        # KPIs
        st.subheader("Missing Value Metrics")
        cols = st.columns(5)
        cols[0].metric("Total Applicants", f"{df_filtered.shape[0]:,}")
        cols[1].metric("Total required inputs", f"{df_filtered.shape[1]:,}")
        cols[2].metric("Duplicate Applicants", f"{df_filtered.duplicated().sum():,}")
        total_missing = df_filtered.isnull().sum().sum()
        total_cells = df_filtered.shape[0] * df_filtered.shape[1]
        percent_missing = (total_missing / total_cells) * 100
        cols[3].metric("Total Missing Values", f"{total_missing:.2f}")
        cols[4].metric("% Missing Cells", f"{percent_missing:.2f}%")

    # Charts
    st.subheader("Missing Values from Defaulted vs Non-Defaulted Applicants")
    df_filtered["HAS_MISSING"] = df_filtered.isnull().any(axis=1)
    defaulted_with_missing = df_filtered[
                            (df_filtered["TARGET"] == 1) &
                            (df_filtered["HAS_MISSING"] == True)
                            ].shape[0]
    nondefaulted_with_missing = df_filtered[
                                (df_filtered["TARGET"] == 0) &
                                (df_filtered["HAS_MISSING"] == True)
                                ].shape[0]
    total_defaulted = (df_filtered["TARGET"] == 1).sum()
    cols = st.columns(2)
    cols[0].metric("Total Defaulted Applicants ",f"{total_defaulted}")
    cols[1].metric( "Total Non Defaulted Applicants ", (df_filtered["TARGET"] == 0).sum()) 
    cols = st.columns(2)
    cols[0].metric( "Defaulted Applicants with Missing Data", defaulted_with_missing)
    cols[1].metric( "Non-Defaulted Applicants with Missing Data", nondefaulted_with_missing)
    defaulted_missing_pct = (defaulted_with_missing / total_defaulted) * 100 
    nondefaulted_missing_pct = (nondefaulted_with_missing / (df_filtered["TARGET"] == 0).sum()) * 100
    cols = st.columns(2)
    cols[0].metric("Defaulted Applicants with Missing Data %", f"{defaulted_missing_pct:.2f}%")
    cols[1].metric("Non-Defaulted Applicants with Missing Data %", f"{nondefaulted_missing_pct:.2f}%")
    #df_filtered.groupby("HAS_MISSING")["TARGET"].mean() * 100


    missing_df = (
    df_filtered.isnull()
    .sum()
    .reset_index())

    missing_df.columns = ["Column", "Missing_Count"]

    missing_df["Missing_Percentage"] = (
        missing_df["Missing_Count"] /
        len(df_filtered)
    ) * 100

    missing_df = missing_df[
        missing_df["Missing_Count"] > 0
    ].sort_values(
        "Missing_Percentage",
        ascending=False
    )
    fig = px.bar(
    missing_df.head(20),
    x="Missing_Percentage",
    y="Column",
    orientation="h",
    title="Top 20 Columns by Missing Percentage")

    st.plotly_chart(fig, use_container_width=True)

    #Categorize missingness
    def missing_category(x):
        if x == 0:
            return "No Missing"
        elif x <= 10:
            return "1–10%"
        elif x <= 30:
            return "10–30%"
        elif x <= 50:
            return "30–50%"
        else:
            return ">50%"
    missing_df["Missing_Category"] = (
    missing_df["Missing_Percentage"]
    .apply(missing_category))

    fig = px.bar(
    missing_df["Missing_Category"]
    .value_counts()
    .reset_index(),
    x="Missing_Category",
    y="count",
    title="Columns by Missing-Data Category")

    st.plotly_chart(fig, use_container_width=True)
    
  #How many missing values does each applicant have?
    df_filtered["Missing_Count"] = df_filtered.isnull().sum(axis=1)
    fig = px.histogram(
    df_filtered,
    x="Missing_Count",
    nbins=30,
    title="Distribution of Missing Values per Applicant")
    st.plotly_chart(fig, use_container_width=True)


    ##################
    #missing fields are associated with higher default?
    results = []

    for col in df_filtered.columns:
        if col in ["TARGET", "HAS_MISSING", "MISSING_COUNT"]:
            continue

        if df_filtered[col].isnull().sum() == 0:
            continue

        missing_default_rate = (
            df_filtered.loc[
                df_filtered[col].isnull(),
                "TARGET"
            ].mean() * 100
        )

        non_missing_default_rate = (
            df_filtered.loc[
                df_filtered[col].notnull(),
                "TARGET"
            ].mean() * 100
        )

        results.append({
            "Column": col,
            "Missing_Count": df_filtered[col].isnull().sum(),
            "Missing_Default_Rate": missing_default_rate,
            "Non_Missing_Default_Rate": non_missing_default_rate
        })

    missing_risk_df = pd.DataFrame(results)
#calculate difference
    missing_risk_df["Default_Rate_Difference"] = (
    missing_risk_df["Missing_Default_Rate"]
    - missing_risk_df["Non_Missing_Default_Rate"]
)
    st.metric("Top Column by Default Rate Difference", missing_risk_df.sort_values(
    "Default_Rate_Difference",
    ascending=False).head(10).iloc[0]["Column"])






except FileNotFoundError:
    st.error("Dataset file not found. Add `data/application_train.csv`.")
