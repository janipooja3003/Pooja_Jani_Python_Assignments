import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import load_application_train_data
from utils.filters import sidebar_filters, apply_filters

st.set_page_config(page_title="Family and Housing Analysis")
st.title("🏠 Family and Housing Analysis")

try:
    df = load_application_train_data()
    filters = sidebar_filters(df)
    df_filtered = apply_filters(df, filters)

    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
    else:
        df_filtered = df_filtered.copy()

        st.subheader("Family and Housing Summary")
        summary_cols = [
            "CNT_FAM_MEMBERS",
            "CNT_CHILDREN",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
        ]
        st.dataframe(df_filtered[summary_cols].describe(include="all").T)

        left, right = st.columns(2)

        with left:
            family_status_data = (
                df_filtered["NAME_FAMILY_STATUS"]
                .dropna()
                .value_counts()
                .rename_axis("Family Status")
                .reset_index(name="Applicants")
            )
            family_status_fig = px.bar(
                family_status_data,
                x="Family Status",
                y="Applicants",
                title="Applicants by Family Status",
                color="Family Status",
            )
            st.plotly_chart(family_status_fig, use_container_width=True)

            family_size_fig = px.histogram(
                df_filtered,
                x="CNT_FAM_MEMBERS",
                nbins=15,
                title="Family Size Distribution",
                labels={"CNT_FAM_MEMBERS": "Family Members"},
                color_discrete_sequence=["#1f77b4"],
            )
            family_size_fig.update_layout(yaxis_title="Number of People")
            st.plotly_chart(family_size_fig, use_container_width=True)

            children_fig = px.histogram(
                df_filtered,
                x="CNT_CHILDREN",
                nbins=12,
                title="Children Count Distribution",
                labels={"CNT_CHILDREN": "Number of Children"},
                color_discrete_sequence=["#17becf"],
            )
            children_fig.update_layout(yaxis_title="Number of People")
            st.plotly_chart(children_fig, use_container_width=True)

        with right:
            housing_data = (
                df_filtered["NAME_HOUSING_TYPE"]
                .dropna()
                .value_counts()
                .rename_axis("Housing Type")
                .reset_index(name="Applicants")
            )
            housing_fig = px.bar(
                housing_data,
                x="Housing Type",
                y="Applicants",
                title="Applicants by Housing Type",
                color="Housing Type",
            )
            st.plotly_chart(housing_fig, use_container_width=True)

            ownership_data = pd.DataFrame({
                "Ownership": ["Owns Car", "Owns Property"],
                "Applicants": [
                    (df_filtered["FLAG_OWN_CAR"] == "Y").sum(),
                    (df_filtered["FLAG_OWN_REALTY"] == "Y").sum(),
                ],
            })
            ownership_fig = px.bar(
                ownership_data,
                x="Ownership",
                y="Applicants",
                title="Car and Property Ownership",
                color="Ownership",
            )
            st.plotly_chart(ownership_fig, use_container_width=True)

        st.subheader("Default Rate by Family and Housing Characteristics")

        def default_rate_summary(column, label):
            result = (
                df_filtered.groupby(column, dropna=False)["TARGET"]
                .agg(Applicants="count", Default_Rate="mean")
                .reset_index()
                .rename(columns={column: label})
            )
            result["Observed Default Rate %"] = result["Default_Rate"] * 100
            return result

        family_default = default_rate_summary("NAME_FAMILY_STATUS", "Family Status")
        housing_default = default_rate_summary("NAME_HOUSING_TYPE", "Housing Type")

        default_left, default_right = st.columns(2)
        with default_left:
            family_default_fig = px.bar(
                family_default,
                x="Family Status",
                y="Observed Default Rate %",
                title="Default Rate by Family Status",
                color="Observed Default Rate %",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(family_default_fig, use_container_width=True)
            st.dataframe(family_default[["Family Status", "Applicants", "Observed Default Rate %"]])

        with default_right:
            housing_default_fig = px.bar(
                housing_default,
                x="Housing Type",
                y="Observed Default Rate %",
                title="Default Rate by Housing Type",
                color="Observed Default Rate %",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(housing_default_fig, use_container_width=True)
            st.dataframe(housing_default[["Housing Type", "Applicants", "Observed Default Rate %"]])

        st.subheader("Having Children and Marital Status by Default Outcome")
        children_marital = df_filtered[["NAME_FAMILY_STATUS", "CNT_CHILDREN", "TARGET"]].copy()
        children_marital["Marital Status"] = children_marital["NAME_FAMILY_STATUS"].eq("Married").map({True: "Married", False: "Not Married"})
        children_marital["Children Status"] = children_marital["CNT_CHILDREN"].gt(0).map({True: "Has Children", False: "No Children"})
        children_marital["Default Outcome"] = children_marital["TARGET"].map({0: "Non-defaulted", 1: "Defaulted"})

        children_marital_summary = (
            children_marital.groupby(["Marital Status", "Children Status", "Default Outcome"], observed=True)
            .size()
            .reset_index(name="Applicants")
        )
        children_marital_summary["Group"] = (
            children_marital_summary["Marital Status"]
            + " / "
            + children_marital_summary["Children Status"]
        )
        children_marital_fig = px.bar(
            children_marital_summary,
            x="Group",
            y="Applicants",
            color="Default Outcome",
            barmode="group",
            title="Applicants With or Without Children by Marital Status",
            labels={"Group": "Marital Status / Children", "Applicants": "Number of People"},
            color_discrete_map={"Defaulted": "#d62728", "Non-defaulted": "#2ca02c"},
        )
        st.plotly_chart(children_marital_fig, use_container_width=True)

    st.write("Applicants:", len(df_filtered))

    st.write("Family sum:", df_filtered["CNT_FAM_MEMBERS"].sum())

    st.write("Family unique:", df_filtered["CNT_FAM_MEMBERS"].nunique())

    st.write("Missing family:", df_filtered["CNT_FAM_MEMBERS"].isna().sum())
    st.write(df_filtered["CNT_FAM_MEMBERS"].value_counts().sort_index())


except FileNotFoundError:
    st.error("Dataset file not found. Add `data/application_train.csv`.")
