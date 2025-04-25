"""
Module: external_db_ingest

Provides a Dagster asset that connects to the Biggs database,
executes user-defined SQL queries using SQLAlchemy, and returns the results as pandas DataFrames.
"""
import pandas as pd
from sqlalchemy import create_engine

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
    Connects to an external database via SQLAlchemy and executes SQL queries.

    Returns:
        dict: A dictionary mapping query index (e.g., 'query_0') to resulting DataFrame.
    """
    config = context.op_config
    results = {}

    # Build SQLAlchemy connection URI
    uri = (
        f"mysql+pymysql://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['database']}"
    )
    engine = create_engine(uri)
    context.log.info("Connected to external MariaDB database via SQLAlchemy.")

    try:
        for i, query in enumerate(config["queries"]):
            df = pd.read_sql(query, con=engine)
            results[f"query_{i}"] = df
            context.log.info(f"Query {i} executed; retrieved {len(df)} rows.")
    except Exception as e:
        context.log.error(f"Error fetching data: {e}")
        raise
    finally:
        engine.dispose()
        context.log.info("SQLAlchemy engine disposed.")

    return results
