import pandas as pd

def create_features(df: pd.DataFrame, output_filepath: str) -> pd.DataFrame:
    """
    Creates new features from the cleaned data and saves the result.

    This function:
    - Computes shipping delay
    - Computes profit margin ratio
    - Extracts time-based features (year / month / weekday)
    - Flags perfect orders
    - Adds advanced engineered features used by the ML model:
        * is_weekend
        * profit_per_day_scheduled
        * category_late_rate
        * customer_late_rate
    """
    print("Creating features...")

    # 1. Shipping Delay (real - scheduled)
    df['shipping_delay'] = (
        df['days_for_shipping_real'] - df['days_for_shipment_scheduled']
    )

    # 2. Profit Margin Ratio (guard against division by zero)
    df['profit_margin_ratio'] = (
        df['benefit_per_order'] / df['sales_per_customer'].replace(0, 1)
    )

    # 3. Extract Time-Based Features from order_date_dateorders
    df['order_year'] = df['order_date_dateorders'].dt.year
    df['order_month'] = df['order_date_dateorders'].dt.month
    df['order_weekday'] = df['order_date_dateorders'].dt.dayofweek

    # 4. Perfect Order Flag
    df['is_perfect_order'] = (
        (df['late_delivery_risk'] == 0) & (df['benefit_per_order'] > 0)
    ).astype(int)

    # ------------------------------------------------------------------
    # 🔥 NEW ENGINEERED FEATURES (must match train_model.py)
    # ------------------------------------------------------------------

    # 5. is_weekend  (1 if order placed on Saturday or Sunday)
    df['is_weekend'] = df['order_weekday'].isin([5, 6]).astype(int)

    # 6. profit_per_day_scheduled
    #    Use max(1, days_for_shipment_scheduled) to avoid division by zero
    df['profit_per_day_scheduled'] = df['benefit_per_order'] / (
        df['days_for_shipment_scheduled'].replace(0, 1)
    )

    # 7. category_late_rate
    #    Average late_delivery_risk per product category
    category_rate_map = (
        df.groupby('category_name')['late_delivery_risk']
          .mean()
    )
    df['category_late_rate'] = df['category_name'].map(category_rate_map)

    # If for some reason a category is missing in the map (very rare),
    # fill with overall mean late_delivery_risk
    overall_late_mean = df['late_delivery_risk'].mean()
    df['category_late_rate'] = df['category_late_rate'].fillna(overall_late_mean)

    # 8. customer_late_rate
    #    Average late_delivery_risk per customer_id
    #    (captures each customer's historical reliability)
    customer_rate_map = (
        df.groupby('customer_id')['late_delivery_risk']
          .mean()
    )
    df['customer_late_rate'] = df['customer_id'].map(customer_rate_map)
    df['customer_late_rate'] = df['customer_late_rate'].fillna(overall_late_mean)

    # ------------------------------------------------------------------
    # Save the feature-engineered data
    # ------------------------------------------------------------------
    df.to_csv(output_filepath, index=False)
    print(f"Feature-engineered data saved to {output_filepath}")

    return df
