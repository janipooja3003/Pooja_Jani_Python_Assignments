# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------

import streamlit as st
import pandas as pd


def sidebar_filters(df: pd.DataFrame) -> dict:
    filters = {}
    st.sidebar.header("Filters")

    # Contract Type Filter
    contract_types = sorted(df["NAME_CONTRACT_TYPE"].dropna().unique())
    selected_contract = st.sidebar.multiselect(
        "Contract Type",
        contract_types,
        default=contract_types
    )
    filters["NAME_CONTRACT_TYPE"] = selected_contract

    # Gender Filter
    genders = sorted(df["CODE_GENDER"].dropna().unique())
    selected_gender = st.sidebar.multiselect(
        "Gender",
        genders,
        default=genders
    )
    filters["CODE_GENDER"] = selected_gender

    # Income Type Filter
    income_types = sorted(df["NAME_INCOME_TYPE"].dropna().unique())
    selected_income = st.sidebar.multiselect(
        "Income Type",
        income_types,
        default=income_types
    )
    filters["NAME_INCOME_TYPE"] = selected_income

    # Education Filter
    education = sorted(df["NAME_EDUCATION_TYPE"].dropna().unique())
    selected_education = st.sidebar.multiselect(
        "Education",
        education,
        default=education
    )
    filters["NAME_EDUCATION_TYPE"] = selected_education

    # Target Filter
    target_options = st.sidebar.multiselect(
        "Application Status",
        [0, 1],
        default=[0, 1],
        format_func=lambda x: "Default / Payment Problem" if x == 1 else "No Default"
    )
    filters["TARGET"] = target_options


    return filters

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    df_filtered = df.copy()
    
    for column, selected_values in filters.items():
        if selected_values:
            df_filtered = df_filtered[df_filtered[column].isin(selected_values)]
    
    return df_filtered
