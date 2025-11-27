import pandas as pd

def add_customer_segments(features_df_path: str,
                          segments_df_path: str,
                          output_filepath: str) -> pd.DataFrame:
    """
    Merges customer segment labels into the main feature dataset.

    - Loads the feature dataset and customer segments from CSV.
    - Merges on `customer_id` using a left join (keep all orders).
    - Tries to use a segment column (cluster / segment / customer_segment).
    - Ensures there is a `cluster` column in the final output, using -1 for
      customers without a segment.

    Parameters
    ----------
    features_df_path : str
        Path to the CSV with engineered features (features.csv).
    segments_df_path : str
        Path to the CSV with customer segments (customer_segments.csv).
    output_filepath : str
        Path where the merged dataset (final_data_with_segments.csv) is saved.

    Returns
    -------
    pd.DataFrame
        The final merged dataframe with a `cluster` column.
    """
    print("Adding customer segments...")

    # 1. Load data
    features_df = pd.read_csv(features_df_path)
    segments_df = pd.read_csv(segments_df_path)

    # 2. Basic checks for customer_id
    if 'customer_id' not in features_df.columns:
        raise KeyError(
            "The features dataset does not contain a 'customer_id' column. "
            "Cannot merge customer segments."
        )
    if 'customer_id' not in segments_df.columns:
        raise KeyError(
            "The segments dataset does not contain a 'customer_id' column. "
            "Cannot merge customer segments."
        ) 

    # 3. Identify the segment column in the segments_df
    #    Prefer 'cluster', but fall back to common alternatives.
    candidate_segment_cols = ['cluster', 'segment', 'customer_segment']
    segment_col = None

    for col in candidate_segment_cols:
        if col in segments_df.columns:
            segment_col = col
            break

    if segment_col is None:
        print(
            "Warning: No segment column found in segments file "
            "(tried: 'cluster', 'segment', 'customer_segment'). "
            "All customers will be assigned cluster = -1."
        )
        # Create an empty cluster column later after merge
        segments_df['cluster'] = -1
        segment_col = 'cluster'
    else:
        # For clarity, rename whatever segment column we found to 'cluster'
        if segment_col != 'cluster':
            segments_df = segments_df.rename(columns={segment_col: 'cluster'})
            segment_col = 'cluster'

    # 4. Keep only customer_id + cluster from segments to avoid duplicates
    segments_df = segments_df[['customer_id', 'cluster']]

    # 5. Merge segments into main features (left join to keep all orders)
    final_df = pd.merge(
        features_df,
        segments_df,
        on='customer_id',
        how='left'
    )

    # 6. Ensure cluster column exists and fill missing with -1 ("Uncategorized")
    if 'cluster' not in final_df.columns:
        final_df['cluster'] = -1
    else:
        final_df['cluster'] = final_df['cluster'].fillna(-1)

    # 7. Save final merged dataset
    final_df.to_csv(output_filepath, index=False)
    print(f"Final data with segments saved to {output_filepath}")

    return final_df
