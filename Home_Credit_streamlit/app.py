import streamlit as st
import pandas as pd
import plotly.express as px
from utils.feature_engineering import calc_kpis
from utils.preprocessing import load_application_train_data
from utils.filters import sidebar_filters, apply_filters
from utils.feature_engineering import calc_kpis #, top_bottom_summary
from utils.charts import line_chart, bar_chart, scatter_chart

st.set_page_config(
    page_title="Home Credit Analytics",
    page_icon=":MoneyBag:",
    layout="wide",
)

st.title("🏠💰 Home Credit Analytics Dashboard")
st.markdown(
    """
    Welcome to the comprehensive **Home Credit Analytics Dashboard** with **20 detailed analysis pages**.
    
    Use the sidebar navigation to explore different aspects of your sales data:
    - **Executive & Overview Pages**: Overall business performance snapshots
    - **Dimensional Analysis**: Deep dives into miising values,income,employment, and family housing
    - **Credit & Risk Analysis**: Credit affordability, default risk, and bureau data insights
    - **Customer & Segment Analysis**: Understand applicants behavior and segmentation
    - **Financial Analysis**: Credit card, POS cash, and installment payment trends
    - **Visualizations**: Interactive charts and graphs for better insights
    """
)

with st.expander("📋 Dashboard Overview"):
    st.write("""
    **20 Pages Available:**
    
    1. **Executive Overview** - High-level business KPIs and trends
    2. **Data Quality** - the data is correct, complete, consistent, and usable for analysis and prediction.
    3. **Missing Value Analysis** - Identify and visualize missing data patterns
    4. **Outlier Analysis** - Detect and visualize outliers in key metrics
    5. **Customer_DemographicsAnalysis** - Demographic breakdown of customers and their impact on sales
    6. **Income Analysis** - Income distribution and its correlation with sales and defaults
    7. **Employment Analysis** - Employment status and its effect on creditworthiness
    8. **Family Housing Analysis** - Family and housing influence on credit risk
    9. **Loan Application Analysis** - Individual applicant performance
    10. **Credit_Affordability Analysis** - Credit affordability and repayment trends
    11. **Default Risk EDA Analysis** - Data Analysis of default risk factors
    12. **Risk Factor Analysis** - Identification of key risk factors affecting credit defaults
    13. **Bureau Credit History Analysis** - Analysis of credit history data from external bureaus
    14. **Bureau Balance Analysis** - Analysis of balance and credit utilization from bureau data
    15. **Previous Applications Analysis** - Analysis of previous loan applications and their outcomes
    16. **POS Cash Analysis** - Point of Sale cash loan analysis and repayment trends
    17. **Installment payment Analysis** - Analysis of installment payments and their impact on credit risk
    18. **Credit Card Analysis** - Analysis of credit card usage, limits, and repayment behavior
    19. **Customer Risk Segmentation** - Segmentation of customers based on risk profiles and creditworthiness
    20. **20_Executive Insights Recommendations** - Actionable insights and recommendations for improving credit risk management and business performance
    """)

with st.expander("📊 Dataset and Instructions"):
    st.write(
        """
        **To use this dashboard:**
        1. Place your CSV file at: `Home_Credit_dashboard/data/application_train.csv`
        2. Use the sidebar filters available on all pages to refine your analysis
        3. Navigate between pages using the Streamlit page menu on the left
        
        **Expected columns in your dataset:**
        
        """
    )

if st.button("🔄 Reload Data"):
    st.rerun()

st.header("Executive Summary")

df = load_application_train_data()
filters = sidebar_filters(df)
df_filtered = apply_filters(df, filters)

def top_defaulted_parameters(data, columns, top_n=10):
            rows = []
            for column in columns:
                if column not in data.columns:
                    continue

                if pd.api.types.is_numeric_dtype(data[column]):
                    numeric_data = data[[column, "TARGET"]].dropna()
                    if numeric_data.empty:
                        continue

                    unique_count = numeric_data[column].nunique()
                    if unique_count < 2:
                        continue

                    bin_count = min(4, unique_count)
                    bins = pd.qcut(
                        numeric_data[column],
                        q=bin_count,
                        duplicates="drop",
                    )
                    summary = (
                        numeric_data.assign(Band=bins)
                        .groupby("Band", observed=True)["TARGET"]
                        .agg(Applicants="count", Default_Rate="mean")
                        .reset_index()
                    )
                    summary["Band"] = summary["Band"].astype(str)
                    top = summary.sort_values("Default_Rate", ascending=False).head(1)
                    if top.empty:
                        continue
                    rows.append(
                        {
                            "Parameter": column,
                            "Top_Value": top.iloc[0]["Band"],
                            "Applicants": int(top.iloc[0]["Applicants"]),
                            "Default_Rate_%": round(top.iloc[0]["Default_Rate"] * 100, 2),
                        }
                    )
                else:
                    summary = (
                        data.groupby(column, dropna=False)["TARGET"]
                        .agg(Applicants="count", Default_Rate="mean")
                        .reset_index()
                    )
                    top = summary.sort_values("Default_Rate", ascending=False).head(1)
                    if top.empty:
                        continue
                    rows.append(
                        {
                            "Parameter": column,
                            "Top_Value": str(top.iloc[0][column]),
                            "Applicants": int(top.iloc[0]["Applicants"]),
                            "Default_Rate_%": round(top.iloc[0]["Default_Rate"] * 100, 2),
                        }
                    )

            if not rows:
                return pd.DataFrame(columns=["Parameter", "Top_Value", "Applicants", "Default_Rate_%"])

            return (
                pd.DataFrame(rows)
                .sort_values("Default_Rate_%", ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )

top_defaulted_cols = [
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "ORGANIZATION_TYPE",
    "OCCUPATION_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

top_defaulted = top_defaulted_parameters(df_filtered, top_defaulted_cols, top_n=10)

if not top_defaulted.empty:
    st.subheader("Top Defaulted Parameters")
    defaulted_metric, _ = st.columns([1, 2])
    with defaulted_metric:
        top_row = top_defaulted.iloc[0]
        st.metric(
            "Highest Default Risk Parameter",
            top_row["Parameter"],
            f"{top_row['Default_Rate_%']:.2f}% default rate",
        )
    top_defaulted_fig = px.bar(
        top_defaulted,
        x="Parameter",
        y="Default_Rate_%",
        title="Top Defaulted Parameters by Observed Default Rate",
        color="Default_Rate_%",
        color_continuous_scale="Sunsetdark",
        text="Top_Value",
    )
    top_defaulted_fig.update_traces(textposition="outside")
    st.plotly_chart(top_defaulted_fig, use_container_width=True)
    st.dataframe(
        top_defaulted[["Parameter", "Top_Value", "Applicants", "Default_Rate_%"]]
        .rename(columns={"Default_Rate_%": "Observed Default Rate %"})   )
