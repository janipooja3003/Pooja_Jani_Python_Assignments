import pandas as pd


def calc_kpis(df: pd.DataFrame) -> dict:
    """Calculate key performance indicators (KPIs) for the filtered dataset"""
    filtered_df = df.copy()

    Application = len(filtered_df)
    default_rate = filtered_df["TARGET"].mean() * 100
    avg_income = filtered_df["AMT_INCOME_TOTAL"].mean()
    avg_credit = filtered_df["AMT_CREDIT"].mean()
    max_credit = filtered_df["AMT_CREDIT"].max()

    return {
        "total_applications": Application,
        "default_rate": default_rate,
        "avg_income": avg_income,
        "avg_credit": avg_credit,
        "max_credit": max_credit
    }

    