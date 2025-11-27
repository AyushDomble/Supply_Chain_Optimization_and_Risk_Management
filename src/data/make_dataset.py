import pandas as pd

def clean_data(input_filepath: str, output_filepath: str) -> pd.DataFrame:
    """
    Loads raw data, standardizes columns, cleans it, and saves the result.

    Steps:
    - Load CSV with latin1 encoding (matches original dataset)
    - Standardize column names: lower-case, underscores, remove brackets
    - Drop PII / unused columns
    - Fix data types and handle basic missing values
    """
    print("Cleaning data...")

    # 1. Load the dataset
    df = pd.read_csv(input_filepath, encoding='latin1')

    # 2. Standardize column names
    #    Use regex=False for parentheses to avoid FutureWarning in pandas
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('(', '', regex=False)
        .str.replace(')', '', regex=False)
    )

    # 3. Drop PII and unnecessary columns (ignore missing)
    columns_to_drop = [
        'customer_email', 'customer_fname', 'customer_lname',
        'customer_password', 'customer_street', 'product_description'
    ]
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    # 4. Clean data types and handle missing values

    # 4.1 Order date
    if 'order_date_dateorders' in df.columns:
        df['order_date_dateorders'] = pd.to_datetime(
            df['order_date_dateorders'],
            errors='coerce'  # invalid dates become NaT instead of crashing
        )
        # If there are any NaT values, you could choose to drop them or fill.
        # For now, we just leave them as NaT and let later steps decide.
    else:
        print("Warning: 'order_date_dateorders' column not found in the dataset.")

    # 4.2 Customer zipcode (only if column exists)
    if 'customer_zipcode' in df.columns:
        df['customer_zipcode'] = df['customer_zipcode'].fillna(0)
    else:
        print("Info: 'customer_zipcode' column not found. Skipping ZIP cleanup.")

    # 5. Save the cleaned data
    df.to_csv(output_filepath, index=False)
    print(f"Cleaned data saved to {output_filepath}")

    return df
