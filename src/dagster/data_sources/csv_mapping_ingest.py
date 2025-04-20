"""
Module: csv_mapping_ingest

Provides a Dagster asset that loads department and item category mapping data
from local CSV files and returns them as pandas DataFrames.
"""

import os
import pandas as pd
from dagster import asset, Field, String

@asset(config_schema={
    "department_filepath": Field(
        String,
        default_value="/dagster/csv/department_category_mapping.csv",  # ✅ absolute inside container
        description="Path to the department category mapping CSV file"
    ),
    "item_filepath": Field(
        String,
        default_value="/dagster/csv/item_category_mapping.csv",  # ✅ absolute inside container
        description="Path to the item category mapping CSV file"
    ),
})
def load_csv_data(context) -> dict:
    """
    Loads department and item mapping data from CSV files with robust path handling.

    Returns:
        dict: {
            "department": department_df,
            "item": item_df
        }
    """
    config = context.op_config

    # Resolve full paths based on Dagster working directory
    cwd = os.getcwd()
    department_path = os.path.join(cwd, config["department_filepath"])
    item_path = os.path.join(cwd, config["item_filepath"])

    # Logging for debugging
    context.log.info(f"📌 Dagster Working Directory: {cwd}")
    context.log.info(f"📄 Department CSV Path: {department_path}")
    context.log.info(f"📄 Item CSV Path: {item_path}")

    # Check existence and load
    if not os.path.exists(department_path):
        raise FileNotFoundError(f"❌ Department CSV not found at: {department_path}")
    if not os.path.exists(item_path):
        raise FileNotFoundError(f"❌ Item CSV not found at: {item_path}")

    department_df = pd.read_csv(department_path)
    item_df = pd.read_csv(item_path)

    context.log.info(f"✅ Loaded department CSV: {department_df.shape}")
    context.log.info(f"✅ Loaded item CSV: {item_df.shape}")

    return {"department": department_df, "item": item_df}