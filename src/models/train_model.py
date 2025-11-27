import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def train_and_save_pipeline(df: pd.DataFrame, model_output_filepath: str) -> None:
    """
    Trains a delivery risk prediction pipeline using XGBoost and GridSearchCV,
    incorporating engineered features, and saves the best model.

    Parameters
    ----------
    df : pd.DataFrame
        The final feature dataset (e.g. final_data_with_segments.csv),
        containing both original and engineered features.
    model_output_filepath : str
        Path where the trained pipeline will be saved (e.g. models/delivery_risk_pipeline.joblib).

    This function:
    - Selects the set of features and the target column
    - Splits into train / test sets with stratification
    - Builds a preprocessing + XGBoost pipeline
    - Runs GridSearchCV to optimize hyperparameters
    - Evaluates the best model (Accuracy, F1, ROC-AUC, Confusion Matrix)
    - Saves the best pipeline to disk
    """
    print("Training XGBoost model pipeline with GridSearchCV...")

    # ----------------------------------------------------------------------
    # 1. Define target and feature set
    # ----------------------------------------------------------------------
    target_col = 'late_delivery_risk'

    # Features expected based on our engineered pipeline
    # (build_features.py + app.py build_feature_row)
    features = [
        # Original Features
        'days_for_shipment_scheduled', 'benefit_per_order', 'sales_per_customer',
        'category_name', 'customer_segment', 'market', 'order_region',
        'shipping_mode', 'order_month', 'order_weekday',

        # Engineered Features
        'is_weekend',
        'profit_per_day_scheduled',
        'category_late_rate',
        'customer_late_rate',
    ]

    # Verify the target exists
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in dataframe. "
            "Please ensure your data contains late_delivery_risk."
        )

    # Check which features are actually present
    existing_features = [f for f in features if f in df.columns]
    missing_features = list(set(features) - set(existing_features))

    if missing_features:
        print(
            "Warning: The following expected features are missing from the input data "
            f"and will NOT be used for training: {missing_features}"
        )

    if not existing_features:
        raise ValueError(
            "No valid training features found in the dataframe. "
            "Please check that feature engineering has been applied correctly."
        )

    print(f"Using {len(existing_features)} features for training.")
    print(f"Feature list: {existing_features}")

    X = df[existing_features]
    y = df[target_col]

    # ----------------------------------------------------------------------
    # 2. Train / Test split
    # ----------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ----------------------------------------------------------------------
    # 3. Preprocessing: numerical vs categorical
    # ----------------------------------------------------------------------
    categorical_features = [
        'category_name', 'customer_segment', 'market',
        'order_region', 'shipping_mode'
    ]

    numerical_features = [
        'days_for_shipment_scheduled', 'benefit_per_order', 'sales_per_customer',
        'order_month', 'order_weekday',
        'is_weekend', 'profit_per_day_scheduled',
        'category_late_rate', 'customer_late_rate'
    ]

    # Filter to only columns that exist in X
    categorical_features = [c for c in categorical_features if c in X.columns]
    numerical_features = [n for n in numerical_features if n in X.columns]

    print(f"Numerical features used: {numerical_features}")
    print(f"Categorical features used: {categorical_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ]
    )

    # ----------------------------------------------------------------------
    # 4. Handle class imbalance: scale_pos_weight
    # ----------------------------------------------------------------------
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    print(f"Class distribution in y_train: positives={pos_count}, negatives={neg_count}")
    print(f"Using scale_pos_weight={scale_pos_weight:.4f}")

    # ----------------------------------------------------------------------
    # 5. Define the model and pipeline
    # ----------------------------------------------------------------------
    xgb_clf = XGBClassifier(
        objective='binary:logistic',
        n_jobs=1,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_clf),
    ])

    # ----------------------------------------------------------------------
    # 6. Grid Search configuration
    # ----------------------------------------------------------------------
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 6],
        'classifier__learning_rate': [0.1, 0.2],
    }

    print("Starting GridSearchCV (this may take a while)...")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',  # balance precision / recall
        n_jobs=1,              # use all cores; change to 1 on Windows if needed
        verbose=1
    )

    # ----------------------------------------------------------------------
    # 7. Fit the model (perform hyperparameter search)
    # ----------------------------------------------------------------------
    grid_search.fit(X_train, y_train)

    print(f"\nGrid Search complete. Best F1 (CV): {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")

    best_pipeline = grid_search.best_estimator_

    # ----------------------------------------------------------------------
    # 8. Evaluation on Test Set
    # ----------------------------------------------------------------------
    print("\nEvaluating best model on Test Set...")

    # Predictions
    y_pred = best_pipeline.predict(X_test)

    # Basic metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Test Set Accuracy: {acc * 100:.2f}%")
    print(f"Test Set F1 Score (weighted): {f1:.4f}")

    # ROC-AUC (requires probabilities)
    try:
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"Test Set ROC-AUC: {auc:.4f}")
    except Exception as e:
        print(f"Could not compute ROC-AUC (reason: {e})")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
 
    # ----------------------------------------------------------------------
    # 9. Save the best pipeline
    # ----------------------------------------------------------------------
    joblib.dump(best_pipeline, model_output_filepath)
    print(f"\nBest model pipeline saved to {model_output_filepath}")
