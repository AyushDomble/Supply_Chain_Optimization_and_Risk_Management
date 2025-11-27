from pathlib import Path

from src.data.make_dataset import clean_data
from src.features.build_features import create_features
from src.features.add_segments import add_customer_segments  # optional enrichment
from src.models.train_model import train_and_save_pipeline


# --- PROJECT ROOT & PATHS ---
# Assumes run_pipeline.py is in the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_orders.csv"
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
SEGMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "customer_segments.csv"
FINAL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_data_with_segments.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "delivery_risk_pipeline.joblib"


if __name__ == "__main__":
    print("\n=== SUPPLY CHAIN PIPELINE START ===\n")

    # Step 1: Clean the raw data
    print(f"Step 1/4: Cleaning raw data from {RAW_DATA_PATH}")
    cleaned_df = clean_data(RAW_DATA_PATH, CLEANED_DATA_PATH)

    # Step 2: Create features from cleaned data
    print(f"\nStep 2/4: Creating features -> {FEATURES_DATA_PATH}")
    features_df = create_features(cleaned_df, FEATURES_DATA_PATH)

    # Step 3: Optionally add customer segments if the file exists
    print("\nStep 3/4: Adding customer segments (if available)")
    if SEGMENTS_PATH.exists():
        print(f"Found segments file at {SEGMENTS_PATH}. Merging segments...")
        final_df = add_customer_segments(
            FEATURES_DATA_PATH,
            SEGMENTS_PATH,
            FINAL_DATA_PATH
        )
    else:
        print(
            f"Warning: {SEGMENTS_PATH} not found.\n"
            "Skipping segment merge. The final data will NOT contain 'cluster' "
            "segment labels, but model training will continue."
        )
        # Ensure FINAL_DATA_PATH still exists for the Streamlit app
        features_df.to_csv(FINAL_DATA_PATH, index=False)
        final_df = features_df

    # Step 4: Train the model on the final, enriched data
    print(f"\nStep 4/4: Training model and saving to {MODEL_PATH}")
    train_and_save_pipeline(final_df, MODEL_PATH)

    print("\n--- Pipeline execution complete! ---")
    print(f"- Cleaned data:        {CLEANED_DATA_PATH}")
    print(f"- Feature data:        {FEATURES_DATA_PATH}")
    print(f"- Final data (app):    {FINAL_DATA_PATH}")
    print(f"- Model pipeline:      {MODEL_PATH}")
    print("\n=== SUPPLY CHAIN PIPELINE END ===\n")
