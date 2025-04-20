# biggs_job.py — modular Dagster pipeline for ML training

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from data_preprocessing.transaction_clean_merge import run_full_preprocessing
from data_sources.csv_mapping_ingest import load_csv_data
from data_sources.external_db_ingest import external_data
from evidently.metric_preset import RegressionPreset
from evidently.report import Report
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from dagster import Definitions, OpExecutionContext, asset, job, op

MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---------------------------
# Assets
# ---------------------------


@asset
def biggs_dataset(external_data: dict, load_csv_data: dict) -> pd.DataFrame:
    """
    Combines and preprocesses external and mapping data to produce enriched transaction dataset.
    """
    return run_full_preprocessing(external_data, load_csv_data)


@asset
def split_data(biggs_dataset: pd.DataFrame):
    """
    Splits the dataset into training and test sets.
    """
    X = biggs_dataset.drop(columns=["target"], errors="ignore")
    y = (
        biggs_dataset["target"]
        if "target" in biggs_dataset.columns
        else np.random.rand(len(X))
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


# ---------------------------
# Utility
# ---------------------------


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


# ---------------------------
# Operations
# ---------------------------


@op(config_schema={"algorithm": str})
def train_model(context, split_data):
    algo = context.op_config["algorithm"]
    X_train = encode_categorical_columns(split_data["X_train"])
    y_train = split_data["y_train"]

    if algo == "xgboost":
        model = XGBRegressor(
            n_estimators=100, random_state=42, objective="reg:squarederror"
        )
    elif algo == "lightgbm":
        model = LGBMRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("Unsupported algorithm")

    model.fit(X_train, y_train)
    return model, algo


@op
def predict(_, model_tuple, split_data):
    model, _ = model_tuple
    X_test = encode_categorical_columns(split_data["X_test"])
    y_pred = model.predict(X_test)
    return {"y_test_pred": y_pred}


@op
def log_to_mlflow(context: OpExecutionContext, model_tuple, split_data, predict):
    model, algo = model_tuple
    y_pred = predict["y_test_pred"]
    X_train = encode_categorical_columns(split_data["X_train"])
    y_train = split_data["y_train"]
    X_test = encode_categorical_columns(split_data["X_test"])
    y_test = split_data["y_test"]

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, f"biggs_{algo}_model")
        mlflow.log_params({"n_estimators": 100, "random_state": 42})
        mlflow.log_metric("train_score", model.score(X_train, y_train))
        mlflow.log_metric("test_score", model.score(X_test, y_test))

        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)
        plt.figure()
        shap.summary_plot(shap_values, X_train, show=False)
        plt.savefig("shap_summary.png")
        mlflow.log_artifact("shap_summary.png")
        plt.close()

        report_data = pd.DataFrame({"target": y_test, "prediction": y_pred})
        report = Report(metrics=[RegressionPreset()])
        report.run(reference_data=report_data, current_data=report_data)
        report.save_html("evidently_report.html")
        mlflow.log_artifact("evidently_report.html")


# ---------------------------
# Print Ops
# ---------------------------


@op
def print_external_data_head(context, ext_data: dict):
    for key, df in ext_data.items():
        context.log.info(f"[external_data: {key}]\n{df.head().to_string()}")


@op
def print_csv_data_head(context, csv_data: dict):
    for key, df in csv_data.items():
        context.log.info(f"[csv_data: {key}]\n{df.head().to_string()}")


@op
def print_biggs_dataset_head(context, df: pd.DataFrame):
    context.log.info("[biggs_dataset]\n" + df.head().to_string())


# ---------------------------
# Job
# ---------------------------


@job
def biggs_training_job():
    ext = external_data()
    csvs = load_csv_data()

    print_external_data_head(ext)
    print_csv_data_head(csvs)

    dataset = biggs_dataset(ext, csvs)
    print_biggs_dataset_head(dataset)

    splits = split_data(dataset)
    model = train_model(splits)
    preds = predict(model, splits)
    log_to_mlflow(model, splits, preds)


# ---------------------------
# Definitions
# ---------------------------

defs = Definitions(
    assets=[external_data, load_csv_data, biggs_dataset, split_data],
    jobs=[biggs_training_job],
)
