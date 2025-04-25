# biggs_job.py — modular Dagster pipeline for ML training

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from data_preprocessing.transaction_clean_merge import run_full_preprocessing
from data_sources.csv_mapping_ingest import load_csv_data
from src.dagster.data_sources.failed_external_db_ingest import load_external_data
from evidently.metric_preset import RegressionPreset
from evidently.report import Report
from lightgbm import LGBMRegressor
from loguru import logger
from pydantic import BaseModel
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple

from dagster import AssetExecutionContext, Definitions, OpExecutionContext, asset, job, op


MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---------------------------
# Assets
# ---------------------------


@asset
def load_and_process_transaction_data(context: AssetExecutionContext, load_external_data: dict, load_csv_data: dict) -> pd.DataFrame:
    """
    Combines and preprocesses external and mapping data to produce enriched transaction dataset.
    """
    df = run_full_preprocessing(load_external_data, load_csv_data)
    head_str = df.head().to_string(index=False)
    context.log.info(f"DataFrame preview:\n{head_str}")
    return df


# @asset
# def split_data(biggs_dataset: pd.DataFrame):
#     """
#     Splits the dataset into training and test sets.
#     """
#     X = biggs_dataset.drop(columns=["target"], errors="ignore")
#     y = (
#         biggs_dataset["target"]
#         if "target" in biggs_dataset.columns
#         else np.random.rand(len(X))
#     )
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#     return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


# ---------------------------
# Utility
# ---------------------------


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


@op
def aggregate_transactions_to_daily(df_transactions: pd.DataFrame) -> pd.DataFrame:
    df_transactions_daily = (
        df_transactions
        .groupby(
            ['branch', 'ite_desc_std', 'date'],
            as_index=False
        )
        .agg({'quantity': 'sum'})
    )
    return df_transactions_daily


@op
def filter_top_10_items_per_branch(
    df_transactions_daily: pd.DataFrame,
    start_year: int = 2023,
    end_year: int = 2024,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Filters df_transactions_daily to the given year range and returns only the top N items
    by total quantity per branch.

    Args:
        df_transactions_daily: Input DataFrame with columns ['branch', 'ite_desc_std', 'date', 'quantity'].
        start_year: First year to include (inclusive).
        end_year: Last year to include (inclusive).
        top_n: Number of top items to keep per branch.

    Returns:
        A DataFrame containing only the top N items (by summed quantity) per branch for the specified years.
    """
    # 1. Restrict to the year window
    df_23_24 = df_transactions_daily[
        (df_transactions_daily['date'].dt.year >= start_year) &
        (df_transactions_daily['date'].dt.year <= end_year)
    ]

    # 2. Compute the top N items per branch
    top_items = (
        df_23_24
        .groupby(['branch', 'ite_desc_std'], as_index=False)['quantity']
        .sum()
        .sort_values(['branch', 'quantity'], ascending=[True, False])
        .groupby('branch')
        .head(top_n)
    )

    # 3. Inner join back to the original to keep only those top items
    result = df_transactions_daily.merge(
        top_items[['branch', 'ite_desc_std']],
        on=['branch', 'ite_desc_std'],
        how='inner'
    )

    return result


def detect_anomalies(df: pd.Series, method: str='iqr', k: float=1.5) -> pd.Series:
    if method=='iqr':
        q1, q3 = df.quantile([0.25,0.75])
        iqr = q3 - q1
        return (df < q1 - k*iqr) | (df > q3 + k*iqr)
        # return (df > q3 + k*iqr)
    elif method=='z':
        return (df - df.mean()).abs()/df.std() > k
    else:
        raise ValueError("method must be 'iqr' or 'z'")


# @op
# def process_filtered_transactions(
#     df_transaction_daily_filtered: pd.DataFrame
# ) -> pd.DataFrame:
#     """
#     Enriches the filtered daily transactions DataFrame with time-based features,
#     flags anomalies, computes seasonal medians, and applies a cap to outliers.

#     Args:
#         df_transaction_daily_filtered: DataFrame with columns
#             ['branch', 'ite_desc_standardized', 'date', 'quantity'].

#     Returns:
#         A DataFrame with added columns:
#           - dayofweek, month, week
#           - anomaly (bool)
#           - typical (seasonal median)
#           - quantity_cap (quantity with outliers replaced by typical)
#     """
#     # Work on a copy to avoid mutating the original
#     df_transaction_daily_filtered = df_transaction_daily_filtered.copy()

#     # 1. Extract time‐based features
#     df_transaction_daily_filtered['dayofweek'] = df_transaction_daily_filtered['date'].dt.dayofweek
#     df_transaction_daily_filtered['month']     = df_transaction_daily_filtered['date'].dt.month
#     df_transaction_daily_filtered['week']      = df_transaction_daily_filtered['date'].dt.isocalendar().week

#     # 2. Flag anomalies per branch + item using IQR method
#     df_transaction_daily_filtered['anomaly'] = (
#         df_transaction_daily_filtered
#         .groupby(['branch', 'ite_desc_standardized'])['quantity']
#         .transform(lambda x: detect_anomalies(x, method='iqr', k=1.5))
#     )

#     # 3. Compute seasonal (monthly) median quantities
#     seasonal = (
#         df_transaction_daily_filtered
#         .groupby(['branch', 'ite_desc_standardized', 'month'])['quantity']
#         .median()
#         .rename('typical')
#         .reset_index()
#     )

#     # 4. Merge the seasonal typicals back into the main DataFrame
#     df_transaction_daily_filtered = df_transaction_daily_filtered.merge(
#         seasonal,
#         on=['branch', 'ite_desc_standardized', 'month'],
#         how='left'
#     )

#     # 5. Cap outliers: replace quantity with typical when flagged as anomaly
#     df_transaction_daily_filtered['quantity_cap'] = np.where(
#         df_transaction_daily_filtered['anomaly'],
#         df_transaction_daily_filtered['typical'],
#         df_transaction_daily_filtered['quantity']
#     )

#     return df_transaction_daily_filtered


# ---------------------------
# Operations
# ---------------------------

# class ForecastConfig(BaseModel):
#     ARIMA_ORDER: Tuple[int, int, int] = (1, 1, 1)
#     TRAIN_END:     str                = "2024-03-31"
#     DATE_COL:      str                = "date"
#     BRANCH_COL:    str                = "branch"
#     ITEM_COL:      str                = "ite_desc_standardized"
#     QUANT_COL:     str                = "quantity_cap"



def compute_horizon(train_end: pd.Timestamp) -> int:
    """
    Compute the forecast horizon as the number of days until the end of next calendar month.
    """
    next_month_end = train_end + pd.offsets.MonthEnd(1)
    horizon = (next_month_end - train_end).days
    return horizon


def forecast_series(
    train_series: pd.Series,
    horizon: int,
    order: tuple
) -> pd.Series:
    """
    Fit an ARIMA to the training series and return a forecast series for the given horizon.
    """
    model = ARIMA(train_series, order=order)
    fit = model.fit()
    start_date = train_series.index.max() + pd.Timedelta(days=1)
    dates_fc = pd.date_range(start=start_date, periods=horizon, freq='D')
    fc_values = fit.forecast(steps=horizon)
    return pd.Series(fc_values, index=dates_fc, name='forecast')


# @op
# def run_arima_pipeline(
#     context: OpExecutionContext,
#     df_capped: pd.DataFrame,
#     config: ForecastConfig
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Runs one-month-ahead ARIMA forecasting for each branch-item series,
#     taking its parameters from the `config` object.
#     """
#     train_end = config.TRAIN_END
#     horizon = compute_horizon(train_end)

#     all_results = []
#     evaluations = []

#     # Unique branch-item combinations
#     series_keys = df_capped[[config.BRANCH_COL, config.ITEM_COL]].drop_duplicates()

#     for _, row in series_keys.iterrows():
#         branch = row[config.BRANCH_COL]
#         item = row[config.ITEM_COL]

#         # Subset and index by date
#         subset = (
#             df_capped
#             .loc[df_capped[config.BRANCH_COL] == branch]
#             .loc[df_capped[config.ITEM_COL] == item]
#             .sort_values(config.DATE_COL)
#             .set_index(config.DATE_COL)
#         )
#         train = subset.loc[:train_end, config.QUANT_COL]
#         test = subset.loc[train_end:, config.QUANT_COL].iloc[:horizon]

#         if train.empty:
#             continue

#         # Fit and forecast
#         try:
#             fc_series = forecast_series(train, horizon, config.ARIMA_ORDER)
#         except Exception as e:
#             continue

#         # Collect forecasts
#         result_df = pd.DataFrame({
#             config.BRANCH_COL: branch,
#             config.ITEM_COL: item,
#             'forecast': fc_series.values
#         }, index=fc_series.index)
#         all_results.append(result_df)

#         # Evaluate
#         eval_df = pd.DataFrame({
#             'test': test,
#             'forecast': fc_series
#         }).dropna()
#         if not eval_df.empty:
#             mae = mean_absolute_error(eval_df['test'], eval_df['forecast'])
#             mape = mean_absolute_percentage_error(eval_df['test'], eval_df['forecast']) * 100
#             evaluations.append((branch, item, mae, mape))

#     # Combine outputs
#     combined_results = pd.concat(all_results) if all_results else pd.DataFrame()
#     eval_df = (
#         pd.DataFrame(evaluations, columns=[
#             config.BRANCH_COL, config.ITEM_COL, 'mae', 'mape'
#         ])
#     )

#     return combined_results, eval_df


# @op(config_schema={"algorithm": str})
# def train_model(context, split_data):
#     algo = context.op_config["algorithm"]
#     X_train = encode_categorical_columns(split_data["X_train"])
#     y_train = split_data["y_train"]

#     if algo == "xgboost":
#         model = XGBRegressor(
#             n_estimators=100, random_state=42, objective="reg:squarederror"
#         )
#     elif algo == "lightgbm":
#         model = LGBMRegressor(n_estimators=100, random_state=42)
#     else:
#         raise ValueError("Unsupported algorithm")

#     model.fit(X_train, y_train)
#     return model, algo


# @op
# def predict(_, model_tuple, split_data):
#     model, _ = model_tuple
#     X_test = encode_categorical_columns(split_data["X_test"])
#     y_pred = model.predict(X_test)
#     return {"y_test_pred": y_pred}


# @op
# def log_to_mlflow(context: OpExecutionContext, model_tuple, split_data, predict):
#     model, algo = model_tuple
#     y_pred = predict["y_test_pred"]
#     X_train = encode_categorical_columns(split_data["X_train"])
#     y_train = split_data["y_train"]
#     X_test = encode_categorical_columns(split_data["X_test"])
#     y_test = split_data["y_test"]

#     with mlflow.start_run() as run:
#         mlflow.sklearn.log_model(model, f"biggs_{algo}_model")
#         mlflow.log_params({"n_estimators": 100, "random_state": 42})
#         mlflow.log_metric("train_score", model.score(X_train, y_train))
#         mlflow.log_metric("test_score", model.score(X_test, y_test))

#         explainer = shap.Explainer(model, X_train)
#         shap_values = explainer(X_train)
#         plt.figure()
#         shap.summary_plot(shap_values, X_train, show=False)
#         plt.savefig("shap_summary.png")
#         mlflow.log_artifact("shap_summary.png")
#         plt.close()

#         report_data = pd.DataFrame({"target": y_test, "prediction": y_pred})
#         report = Report(metrics=[RegressionPreset()])
#         report.run(reference_data=report_data, current_data=report_data)
#         report.save_html("evidently_report.html")
#         mlflow.log_artifact("evidently_report.html")


# ---------------------------
# Print Ops
# ---------------------------


@op
def print_data_head(context, ext_data: dict):
    for key, df in ext_data.items():
        context.log.info(f"[data: {key}]\n{df.head().to_string()}")


# ---------------------------
# Job
# ---------------------------


@job
def biggs_training_job():
    biggs_db_data = load_external_data()
    biggs_csv_mapping = load_csv_data()

    print_data_head(biggs_db_data)
    print_data_head(biggs_csv_mapping)

    df_transactions = load_and_process_transaction_data(biggs_db_data, biggs_csv_mapping)
    # df_transactions.head()

    df_transaction_daily = aggregate_transactions_to_daily(df_transactions)
    # df_transaction_daily.head()

    df_transaction_daily_filtered = filter_top_10_items_per_branch(df_transaction_daily)
    print_data_head(df_transaction_daily_filtered)

    # df_data_with_anomaly_flags_and_capped_values = process_filtered_transactions(df_transaction_daily_filtered)
    # print_data_head(df_data_with_anomaly_flags_and_capped_values)

    # df_capped_data = df_data_with_anomaly_flags_and_capped_values[
    #     ['branch', 'ite_desc_standardized', 'date', 'quantity_cap']]
    
    # combined_results, eval_df = run_arima_pipeline(df_capped_data)

    # print_data_head(eval_df)
    # print_data_head(combined_results)
    # splits = split_data(df_transactions)
    # model = train_model(splits)
    # preds = predict(model, splits)
    # log_to_mlflow(model, splits, preds)


# ---------------------------
# Definitions
# ---------------------------

defs = Definitions(
    assets=[
        load_external_data, 
        load_csv_data, 
        load_and_process_transaction_data,
    ],
    # assets=[load_external_data, load_csv_data, load_and_process_transaction_data, aggregate_transactions_to_daily, 
    #         filter_top_10_items_per_branch, process_filtered_transactions, run_arima_pipeline, split_data],
    jobs=[biggs_training_job],
)
