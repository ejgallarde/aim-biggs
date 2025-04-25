"""
Module: external_db_ingest

Provides a Dagster asset that connects to the Biggs database,
executes user-defined SQL queries, and returns the results as pandas DataFrames.
"""

import pandas as pd
import pymysql

from dagster import Array, Field, Int, String, asset


@asset(
    config_schema={
        "host": Field(String, description="External DB host (e.g., 'localhost')"),
        "port": Field(Int, description="Port number for the DB (e.g., 3306)"),
        "user": Field(String, description="Database username"),
        "password": Field(String, description="Database password"),
        "database": Field(String, description="Target database name"),
        "queries": Field(Array(String), description="List of SQL queries to execute"),
    }
)
def load_external_data(context) -> dict:
    """
    Connects to an external database and executes SQL queries.

    Returns:
        dict: A dictionary mapping query index (e.g., 'query_0') to resulting DataFrame.
    """
    config = context.op_config
    results = {}
    conn = None

    try:
        conn = pymysql.connect(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            database=config["database"],
        )
        context.log.info("Connected to external MariaDB database.")

        for i, query in enumerate(config["queries"]):
            df = pd.read_sql(query, conn)
            results[f"query_{i}"] = df
            context.log.info(f"Query {i} executed; retrieved {len(df)} rows.")

    except Exception as e:
        context.log.error(f"Error fetching data: {e}")
        raise

    finally:
        if conn:
            conn.close()
            context.log.info("Database connection closed.")

    return results
