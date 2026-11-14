
import pandas as pd
from pathlib import Path
import joblib
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ==========================================
# MODEL FILE LOCATION
# ==========================================

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "demand_model.pkl"


# ==========================================
# 1. PREPARE SALES DATA
# ==========================================

def prepare_sales_data(sales):

    df = pd.DataFrame(sales).copy()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                "product_name",
                "date",
                "quantity_sold"
            ]
        )

    df.columns = df.columns.str.strip()

    required_columns = [
        "product_id",
        "quantity",
        "sold_at"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}. "
                f"Available columns: {df.columns.tolist()}. "
                "Check get_sales() in database.py."
            )

    # Product ID must be the actual product ID,
    # not the sale record ID.
    df["product_id"] = pd.to_numeric(
        df["product_id"],
        errors="coerce"
    )

    df["quantity"] = pd.to_numeric(
        df["quantity"],
        errors="coerce"
    )

    df["sold_at"] = pd.to_datetime(
        df["sold_at"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["product_id", "quantity", "sold_at"]
    )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                "product_name",
                "date",
                "quantity_sold"
            ]
        )

    df["product_id"] = df["product_id"].astype(int)

    # Add product names if they are missing.
    if "product_name" not in df.columns:
        df["product_name"] = (
            "Product " + df["product_id"].astype(str)
        )
    else:
        df["product_name"] = (
            df["product_name"]
            .fillna("")
            .astype(str)
        )

        df.loc[
            df["product_name"].str.strip() == "",
            "product_name"
        ] = (
            "Product "
            + df["product_id"].astype(str)
        )

    # Extract the date without the time.
    df["date"] = df["sold_at"].dt.normalize()

    # Combine sales of the same product on the same day.
    daily_sales = (
        df.groupby(
            ["product_id", "product_name", "date"],
            as_index=False
        )["quantity"]
        .sum()
    )

    daily_sales = daily_sales.rename(
        columns={"quantity": "quantity_sold"}
    )

    # Add dates with no sales and assign zero sales.
    complete_data = []

    for product_id in daily_sales["product_id"].unique():

        product_data = daily_sales[
            daily_sales["product_id"] == product_id
        ].copy()

        product_name = product_data[
            "product_name"
        ].iloc[0]

        date_range = pd.date_range(
            start=product_data["date"].min(),
            end=product_data["date"].max(),
            freq="D"
        )

        product_data = (
            product_data.set_index("date")
            .reindex(date_range)
        )

        product_data.index.name = "date"

        product_data["product_id"] = product_id
        product_data["product_name"] = product_name

        product_data["quantity_sold"] = (
            product_data["quantity_sold"].fillna(0)
        )

        product_data = product_data.reset_index()

        complete_data.append(product_data)

    if not complete_data:
        return pd.DataFrame(
            columns=[
                "product_id",
                "product_name",
                "date",
                "quantity_sold"
            ]
        )

    final_dataset = pd.concat(
        complete_data,
        ignore_index=True
    )

    return final_dataset[
        [
            "product_id",
            "product_name",
            "date",
            "quantity_sold"
        ]
    ]


# ==========================================
# 2. CREATE MACHINE-LEARNING FEATURES
# ==========================================

def create_features(product_data):

    df = product_data.copy()

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["quantity_sold"] = pd.to_numeric(
        df["quantity_sold"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date", "quantity_sold"]
    )

    df = df.sort_values("date").copy()

    # Previous day's sales.
    df["lag_1"] = df["quantity_sold"].shift(1)

    # Sales from seven days earlier.
    df["lag_7"] = df["quantity_sold"].shift(7)

    # Average sales over the previous seven days.
    df["avg_7"] = (
        df["quantity_sold"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    # Calendar information.
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    return df.dropna().copy()


# ==========================================
# 3. TRAIN DEMAND PREDICTION MODELS
# ==========================================

def train_demand_models(dataset):

    feature_columns = [
        "lag_1",
        "lag_7",
        "avg_7",
        "day_of_week",
        "month"
    ]

    trained_models = {}
    model_metrics = {}
    skipped_products = []

    if dataset.empty:
        # Remove an old model if no new training data exists.
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()

        return model_metrics, skipped_products

    required_columns = [
        "product_id",
        "product_name",
        "date",
        "quantity_sold"
    ]

    for column in required_columns:
        if column not in dataset.columns:
            raise ValueError(
                f"Missing column in training dataset: {column}"
            )

    for product_id in dataset["product_id"].unique():

        product_data = dataset[
            dataset["product_id"] == product_id
        ].copy()

        product_data["date"] = pd.to_datetime(
            product_data["date"],
            errors="coerce"
        )

        product_data = product_data.dropna(
            subset=["date", "quantity_sold"]
        )

        product_data = product_data.sort_values(
            "date"
        )

        features = create_features(product_data)

        # Require sufficient historical data.
        if len(product_data) < 10 or len(features) < 5:

            if not product_data.empty:
                skipped_products.append(
                    str(product_data["product_name"].iloc[0])
                )

            continue

        X = features[feature_columns]
        y = features["quantity_sold"]

        # Use earlier records for training
        # and later records for testing.
        split_index = int(len(X) * 0.8)

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        if X_train.empty or X_test.empty:
            skipped_products.append(
                str(product_data["product_name"].iloc[0])
            )
            continue

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        # First training: evaluation.
        model.fit(X_train, y_train)

        test_predictions = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            test_predictions
        )

        # Second training: use all available data.
        model.fit(X, y)

        trained_models[int(product_id)] = model

        model_metrics[int(product_id)] = {
            "product_name": str(
                product_data["product_name"].iloc[0]
            ),
            "mae": round(float(mae), 2),
            "training_records": len(X)
        }

    # Save the newly trained models safely.
    # A temporary file prevents a half-written model
    # from replacing the previous saved model.
    if trained_models:

        temp_model_path = MODEL_PATH.with_name(
            "demand_model.tmp"
        )

        try:
            joblib.dump(
                trained_models,
                temp_model_path
            )

            temp_model_path.replace(MODEL_PATH)

        finally:
            if temp_model_path.exists():
                temp_model_path.unlink()

    else:
        # Do not keep an old or corrupted model
        # when this training attempt produced no models.
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()

    return model_metrics, skipped_products


# ==========================================
# 4. LOAD SAVED DEMAND MODELS
# ==========================================

def load_demand_models():

    if not MODEL_PATH.exists():
        return {}

    # Empty model files cannot be loaded.
    if MODEL_PATH.stat().st_size == 0:
        return {}

    try:

        models = joblib.load(MODEL_PATH)

        if not isinstance(models, dict):
            return {}

        return models

    except (
        EOFError,
        ValueError,
        OSError,
        pickle.UnpicklingError,
        AttributeError,
        ImportError
    ):

        # Return an empty dictionary instead of
        # crashing the Streamlit application.
        return {}


# ==========================================
# 5. PREDICT TOMORROW'S DEMAND
# ==========================================

def predict_tomorrow(dataset, product_id):

    if dataset.empty:
        return None

    models = load_demand_models()

    product_id = int(product_id)

    if product_id not in models:
        return None

    product_data = dataset[
        dataset["product_id"] == product_id
    ].sort_values("date").copy()

    if len(product_data) < 7:
        return None

    quantities = product_data[
        "quantity_sold"
    ].astype(float)

    tomorrow = (
        pd.to_datetime(product_data["date"]).max()
        + pd.Timedelta(days=1)
    )

    input_data = pd.DataFrame([{
        "lag_1": quantities.iloc[-1],
        "lag_7": quantities.iloc[-7],
        "avg_7": quantities.tail(7).mean(),
        "day_of_week": tomorrow.dayofweek,
        "month": tomorrow.month
    }])

    model = models[product_id]

    prediction = model.predict(input_data)[0]

    return max(
        0,
        round(float(prediction))
    )


# ==========================================
# 6. SMART STOCK RECOMMENDATION
# ==========================================

def get_stock_recommendation(
    dataset,
    product_id,
    current_stock
):

    if dataset.empty:
        return None

    models = load_demand_models()

    product_id = int(product_id)

    if product_id not in models:
        return None

    product_data = dataset[
        dataset["product_id"] == product_id
    ].sort_values("date").copy()

    if len(product_data) < 7:
        return None

    model = models[product_id]

    # Recent daily sales history.
    history = (
        product_data["quantity_sold"]
        .astype(float)
        .tolist()
    )

    last_date = pd.to_datetime(
        product_data["date"]
    ).max()

    forecast = []

    # Predict demand for the next seven days.
    for day in range(1, 8):

        prediction_date = (
            last_date + pd.Timedelta(days=day)
        )

        input_data = pd.DataFrame([{
            "lag_1": history[-1],
            "lag_7": history[-7],
            "avg_7": sum(history[-7:]) / 7,
            "day_of_week": prediction_date.dayofweek,
            "month": prediction_date.month
        }])

        predicted_quantity = max(
            0,
            float(model.predict(input_data)[0])
        )

        forecast.append(predicted_quantity)

        # Use the predicted sales as history
        # for the following day's prediction.
        history.append(predicted_quantity)

    weekly_demand = sum(forecast)

    # Keep 20% extra stock as a safety buffer.
    safety_stock = round(
        weekly_demand * 0.20
    )

    # Reorder point.
    reorder_point = round(
        weekly_demand + safety_stock
    )

    # Target approximately 14 days of demand
    # plus safety stock.
    target_stock = round(
        (weekly_demand / 7) * 14 + safety_stock
    )

    current_stock = max(
        0,
        int(current_stock)
    )

    suggested_order = max(
        0,
        target_stock - current_stock
    )

    return {
        "weekly_demand": round(weekly_demand),
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "target_stock": target_stock,
        "suggested_order": suggested_order
    }

