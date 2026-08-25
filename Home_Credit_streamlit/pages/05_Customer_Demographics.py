import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import load_application_train_data
from utils.filters import sidebar_filters, apply_filters

st.set_page_config(page_title="Customer Demographics")
st.title("👥 Customer Demographics")

try:
    df = load_application_train_data()
    filters = sidebar_filters(df)
    df_filtered = apply_filters(df, filters)

    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
    else:
        df_filtered = df_filtered.copy()
        df_filtered["AGE_YEARS"] = abs(df_filtered["DAYS_BIRTH"]) / 365.25

        st.subheader("Demographic Summary")
        demo_cols = [
            "CODE_GENDER",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_INCOME_TYPE",
            "CNT_CHILDREN",
            "AGE_YEARS",
        ]

        summary = {}
        for col in demo_cols:
            if col in df_filtered.columns:
                if df_filtered[col].dtype == "object":
                    val_counts = df_filtered[col].dropna().value_counts().reset_index()
                    val_counts.columns = [col, "Count"]
                    summary[col] = val_counts
                else:
                    summary[col] = df_filtered[col].describe()

        left, right = st.columns(2)

        with left:
            gender_data = df_filtered["CODE_GENDER"].value_counts().rename_axis("Gender").reset_index(name="Count")
            gender_fig = px.bar(
                gender_data,
                x="Gender",
                y="Count",
                title="Gender Distribution",
                color="Gender",
            )
            st.plotly_chart(gender_fig, use_container_width=True)

            education_data = df_filtered["NAME_EDUCATION_TYPE"].value_counts().rename_axis("Education").reset_index(name="Count")
            education_fig = px.bar(
                education_data,
                x="Education",
                y="Count",
                title="Education Level Distribution",
                color="Education",
            )
            st.plotly_chart(education_fig, use_container_width=True)

        with right:
            family_data = df_filtered["NAME_FAMILY_STATUS"].value_counts().rename_axis("Family Status").reset_index(name="Count")
            family_fig = px.bar(
                family_data,
                x="Family Status",
                y="Count",
                title="Family Status Distribution",
                color="Family Status",
            )
            st.plotly_chart(family_fig, use_container_width=True)

            housing_data = df_filtered["NAME_HOUSING_TYPE"].value_counts().rename_axis("Housing Type").reset_index(name="Count")
            housing_fig = px.bar(
                housing_data,
                x="Housing Type",
                y="Count",
                title="Housing Type Distribution",
                color="Housing Type",
            )
            st.plotly_chart(housing_fig, use_container_width=True)

        age_fig = px.histogram(
            df_filtered,
            x="AGE_YEARS",
            nbins=40,
            title="Age Distribution",
            labels={"AGE_YEARS": "Age (years)"},
        )
        age_fig.update_xaxes(title_text="Age (years)")
        st.plotly_chart(age_fig, use_container_width=True)

        child_fig = px.histogram(
            df_filtered,
            x="CNT_CHILDREN",
            nbins=10,
            title="Children Count Distribution",
            labels={"CNT_CHILDREN": "Number of Children"},
        )
        st.plotly_chart(child_fig, use_container_width=True)

        st.subheader("Demographic Table")
        st.dataframe(
            df_filtered[["CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "NAME_EDUCATION_TYPE", "CNT_CHILDREN", "AGE_YEARS"]].describe(include="all").T
        )

except FileNotFoundError:
    st.error("Dataset file not found. Add `data/application_train.csv`.")
