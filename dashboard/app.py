import streamlit as st
import pandas as pd
import joblib
import os
import pycountry
import plotly.express as px
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Supply Chain Dashboard",
    page_icon="🚚",
    layout="wide"
)

# --- BUILD ABSOLUTE PATHS ---
PROJECT_ROOT = Path(__file__).parent.parent  # assumes app.py is in a subfolder under project root
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_data_with_segments.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "delivery_risk_pipeline.joblib"


# --- DATA AND MODEL LOADING ---
@st.cache_data
def load_data() -> pd.DataFrame:
    """Loads the final dataset with features and segments."""
    df = pd.read_csv(DATA_PATH, parse_dates=['order_date_dateorders'])
    return df


@st.cache_resource
def load_prediction_pipeline():
    """Loads the saved prediction pipeline."""
    pipeline = joblib.load(MODEL_PATH)
    return pipeline


@st.cache_resource
def load_forecast_models():
    """Loads all regional demand forecast models."""
    models = {}
    models_path = PROJECT_ROOT / "models"
    if not models_path.exists():
        return models

    for filename in os.listdir(models_path):
        if filename.startswith("demand_forecaster_") and filename.endswith(".pkl"):
            region = (
                filename
                .replace("demand_forecaster_", "")
                .replace(".pkl", "")
                .replace("_", " ")
            )
            models[region] = joblib.load(models_path / filename)
    return models


# --- HELPER: BUILD FEATURE ROW FOR MODEL ---
def build_feature_row(
    *,
    order_date,
    days_scheduled: int,
    sales: float,
    benefit: float,
    shipping_mode: str,
    customer_segment: str,
    market: str,
    category: str,
    order_region: str,
    base_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a single-row DataFrame with ALL features required by the model:
    - Core features
    - Engineered features: is_weekend, profit_per_day_scheduled,
      category_late_rate, customer_late_rate
    """
    # --- Core features ---
    order_month = order_date.month
    order_weekday = order_date.weekday()  # 0=Mon ... 6=Sun

    data = {
        'days_for_shipment_scheduled': [days_scheduled],
        'benefit_per_order': [benefit],
        'sales_per_customer': [sales],
        'category_name': [category],
        'customer_segment': [customer_segment],
        'market': [market],
        'order_region': [order_region],
        'shipping_mode': [shipping_mode],
        'order_month': [order_month],
        'order_weekday': [order_weekday],
    }

    df_input = pd.DataFrame(data)

    # --- Engineered: is_weekend ---
    df_input['is_weekend'] = int(order_weekday in [5, 6])

    # --- Engineered: profit_per_day_scheduled ---
    safe_days = days_scheduled if days_scheduled != 0 else 1
    df_input['profit_per_day_scheduled'] = benefit / safe_days

    # --- Engineered: category_late_rate ---
    category_rate = (
        base_df.groupby('category_name')['late_delivery_risk']
        .mean()
    )
    overall_late_mean = base_df['late_delivery_risk'].mean()

    df_input['category_late_rate'] = category_rate.get(category, overall_late_mean)

    # --- Engineered: customer_late_rate ---
    # Approximate customer-level history using segment-level average
    segment_rate = (
        base_df.groupby('customer_segment')['late_delivery_risk']
        .mean()
    )
    df_input['customer_late_rate'] = segment_rate.get(customer_segment, overall_late_mean)

    return df_input


# --- LOAD DATA & MODELS ---
forecast_models = load_forecast_models()
df = load_data()
prediction_pipeline = load_prediction_pipeline()

# Dataset info
min_date = df['order_date_dateorders'].min().date()
max_date = df['order_date_dateorders'].max().date()
st.info(f"Full dataset ranges from {min_date} to {max_date}")

# --- SIDEBAR FILTERS ---
st.sidebar.header("Dashboard Filters")

# Filter by Region
selected_region = st.sidebar.multiselect(
    "Filter by Region",
    options=df['order_region'].unique(),
    default=df['order_region'].unique()
)

# Filter by Shipping Mode
selected_shipping = st.sidebar.multiselect(
    "Filter by Shipping Mode",
    options=df['shipping_mode'].unique(),
    default=df['shipping_mode'].unique()
)

# Date filters
start_date = st.sidebar.date_input("Start Date", df['order_date_dateorders'].min())
end_date = st.sidebar.date_input("End Date", df['order_date_dateorders'].max())

# Apply filters
df_filtered = df[
    (df['order_date_dateorders'] >= pd.to_datetime(start_date)) &
    (df['order_date_dateorders'] <= pd.to_datetime(end_date)) &
    df['order_region'].isin(selected_region) &
    df['shipping_mode'].isin(selected_shipping)
]

# --- HEADER ---
st.title("Supply Chain Optimization & Risk Management Dashboard 🚚")
st.write(
    "This dashboard provides an overview of supply chain performance, "
    "customer segments, and demand forecasts, along with a tool to "
    "predict late delivery risk for new orders."
)

st.markdown("---")

# --- KPIs ---
st.header("Key Performance Indicators")

otif_rate = 1 - df_filtered['late_delivery_risk'].mean()
perfect_order_rate = df_filtered['is_perfect_order'].mean()
avg_real_days = df_filtered['days_for_shipping_real'].mean()
avg_scheduled_days = df_filtered['days_for_shipment_scheduled'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("On-Time-In-Full (OTIF) Rate", f"{otif_rate:.2%}")
col2.metric("Perfect Order Rate", f"{perfect_order_rate:.2%}")
col3.metric("Avg. Real Shipping Days", f"{avg_real_days:.2f}")
col4.metric("Avg. Scheduled Shipping Days", f"{avg_scheduled_days:.2f}")

st.markdown("---")

# --- DATA EXPORTER ---
@st.cache_data
def convert_df_to_csv(df_to_export: pd.DataFrame) -> bytes:
    """Converts a DataFrame to a CSV string."""
    return df_to_export.to_csv(index=False).encode('utf-8')


csv_data = convert_df_to_csv(df_filtered)

st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv_data,
    file_name='filtered_supply_chain_data.csv',
    mime='text/csv',
)

# --- ANALYTICS CHARTS ---
st.header("Supply Chain Analytics")

fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.subheader("Late Delivery Risk by Shipping Mode")

    risk_by_shipping_mode = (
        df_filtered.groupby('shipping_mode')['late_delivery_risk']
        .mean()
        .sort_values(ascending=False)
    )

    fig = px.bar(
        risk_by_shipping_mode,
        x=risk_by_shipping_mode.index,
        y='late_delivery_risk',
        title="Late Delivery Risk per Shipping Mode",
        labels={'late_delivery_risk': 'Late Risk %', 'shipping_mode': 'Shipping Mode'},
        color='late_delivery_risk',
        color_continuous_scale=px.colors.sequential.YlOrRd
    )
    fig.update_layout(yaxis_tickformat='.2%')
    st.plotly_chart(fig, use_container_width=True)

with fig_col2:
    st.subheader("Geographical Late Delivery Risk")

    risk_by_country = (
        df_filtered.groupby('customer_country')['late_delivery_risk']
        .mean()
        .reset_index()
    )

    def get_iso_alpha(country_name: str):
        try:
            return pycountry.countries.get(name=country_name).alpha_3
        except AttributeError:
            if country_name == "EE. UU.":  # Spanish for USA
                return "USA"
            return None

    risk_by_country['iso_alpha'] = risk_by_country['customer_country'].apply(get_iso_alpha)

    fig = px.choropleth(
        risk_by_country.dropna(subset=['iso_alpha']),
        locations="iso_alpha",
        locationmode='ISO-3',
        color="late_delivery_risk",
        hover_name="customer_country",
        color_continuous_scale=px.colors.sequential.YlOrRd,
        title="Late Delivery Risk % by Country"
    )
    fig.update_layout(geo=dict(showcoastlines=True))
    st.plotly_chart(fig, use_container_width=True)

# --- CATEGORY PROFITABILITY VS RISK ---
st.markdown("---")
st.header("Category Performance")
st.subheader("Profitability vs. Risk Scatter Plot")

category_analysis = df_filtered.groupby('category_name').agg(
    total_profit=('benefit_per_order', 'sum'),
    avg_late_risk=('late_delivery_risk', 'mean')
).reset_index()

category_analysis['size_scaled'] = category_analysis['total_profit'].clip(lower=0)

fig_scatter = px.scatter(
    category_analysis,
    x="total_profit",
    y="avg_late_risk",
    size="size_scaled",
    color="avg_late_risk",
    hover_name="category_name",
    color_continuous_scale=px.colors.sequential.YlOrRd,
    labels={
        "total_profit": "Total Profit ($)",
        "avg_late_risk": "Average Late Delivery Risk"
    },
    title="Product Category Performance: Profit vs. Risk"
)
fig_scatter.update_layout(yaxis_tickformat='.2%')
st.plotly_chart(fig_scatter, use_container_width=True)

# --- DEMAND FORECAST ---
st.markdown("---")
st.header("Demand Forecast by Region")

if forecast_models:
    selected_forecast_region = st.selectbox(
        "Select a Region to Forecast",
        options=list(forecast_models.keys())
    )

    if selected_forecast_region:
        model = forecast_models[selected_forecast_region]
        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)

        st.subheader(f"90-Day Demand Forecast for {selected_forecast_region}")
        fig_forecast = model.plot(forecast)
        st.pyplot(fig_forecast)
else:
    st.info("No demand forecast models were found in the models folder.")

# --- ML PREDICTION TOOL ---
st.sidebar.markdown("---")
st.sidebar.header("📦 Predict Late Delivery Risk")
st.sidebar.write("Enter new order details to get a risk prediction.")

order_date = st.sidebar.date_input("Order Date")
days_scheduled = st.sidebar.slider("Days for Shipment (Scheduled)", min_value=0, max_value=10, value=4)
sales = st.sidebar.number_input("Sales per Order ($)", min_value=0.0, value=250.0)
benefit = st.sidebar.number_input("Benefit per Order ($)", value=50.0)

shipping_mode = st.sidebar.selectbox("Shipping Mode", options=df['shipping_mode'].unique())
customer_segment = st.sidebar.selectbox("Customer Segment", options=df['customer_segment'].unique())
market = st.sidebar.selectbox("Market", options=df['market'].unique())
category = st.sidebar.selectbox("Product Category", options=df['category_name'].unique())
order_region = st.sidebar.selectbox("Order Region", options=df['order_region'].unique())

# --- PREDICT RISK BUTTON ---
if st.sidebar.button("Predict Risk"):
    # Build full feature row (core + engineered)
    input_df = build_feature_row(
        order_date=order_date,
        days_scheduled=days_scheduled,
        sales=sales,
        benefit=benefit,
        shipping_mode=shipping_mode,
        customer_segment=customer_segment,
        market=market,
        category=category,
        order_region=order_region,
        base_df=df
    )

    prediction_proba = prediction_pipeline.predict_proba(input_df)[0][1]  # P(class=1)
    risk_percent = prediction_proba * 100

    st.sidebar.subheader("Prediction Result")

    if risk_percent >= 50:
        st.sidebar.metric(
            label="Risk of Late Delivery",
            value=f"{risk_percent:.2f}%",
            delta="High Risk",
            delta_color="inverse"
        )
    else:
        st.sidebar.metric(
            label="Risk of Late Delivery",
            value=f"{risk_percent:.2f}%",
            delta="Low Risk",
            delta_color="normal"
        )

    # --- Top Factors Influencing Prediction ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Top Factors Influencing Prediction:")

    feature_importances = prediction_pipeline.named_steps['classifier'].feature_importances_
    feature_names = prediction_pipeline.named_steps['preprocessor'].get_feature_names_out()

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values(by='importance', ascending=False)

    # Ensure importance is a plain float type for JSON serialization
    importance_df['importance'] = importance_df['importance'].astype(float)
    max_importance = float(importance_df['importance'].max())

    st.sidebar.dataframe(
        importance_df.head(5),
        column_config={
            "feature": "Factor",
            "importance": st.column_config.ProgressColumn(
                "Importance",
                format="%.3f",
                min_value=0.0,
                max_value=max_importance,
            ),
        },
        hide_index=True
    )

# --- RECOMMEND OPTIMAL SHIPPING MODE ---
if st.sidebar.button("Recommend Optimal Shipping Mode"):
    shipping_mode_profitability = df.groupby('shipping_mode')['benefit_per_order'].mean()

    recommendations = []

    for mode in df['shipping_mode'].unique():
        candidate_df = build_feature_row(
            order_date=order_date,
            days_scheduled=days_scheduled,
            sales=sales,
            benefit=benefit,
            shipping_mode=mode,
            customer_segment=customer_segment,
            market=market,
            category=category,
            order_region=order_region,
            base_df=df
        )

        prediction_proba = prediction_pipeline.predict_proba(candidate_df)[0][1]

        recommendations.append({
            "mode": mode,
            "risk": float(prediction_proba),
            "profitability": float(shipping_mode_profitability.get(mode, 0.0))
        })

    reco_df = pd.DataFrame(recommendations)

    st.sidebar.subheader("Recommendation Result")

    feasible_options = reco_df[reco_df['risk'] < 0.35]

    if not feasible_options.empty:
        best_option = feasible_options.loc[feasible_options['profitability'].idxmax()]
        st.sidebar.success(f"**Recommended Mode:** {best_option['mode']}")
        st.sidebar.write(
            f"This option has a low late delivery risk of **{best_option['risk']:.2%}** "
            f"and good profitability."
        )
    else:
        lowest_risk_option = reco_df.loc[reco_df['risk'].idxmin()]
        st.sidebar.warning("**No low-risk option found under the threshold.**")
        st.sidebar.info(
            f"The safest available option is **{lowest_risk_option['mode']}** "
            f"with a late delivery risk of **{lowest_risk_option['risk']:.2%}**."
        )
