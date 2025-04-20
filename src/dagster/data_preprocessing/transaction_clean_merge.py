# File: data_preprocessing/transaction_clean_merge.py

import pandas as pd
import re
from typing import Dict
from data_sources.external_db_ingest import external_data
from data_sources.csv_mapping_ingest import load_csv_data


def run_full_preprocessing(external: Dict[str, pd.DataFrame], csv_maps: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Orchestrates the full preprocessing pipeline:
    - Cleans transaction data from RD5000
    - Enriches product data using RD5500 and RD1800
    - Applies item and department category mappings
    - Merges enriched product info into transactions
    - Filters the final transaction master

    Parameters:
        external (dict): Output from external_data() containing RD5000, RD5500, and RD1800
        csv_maps (dict): Output from load_csv_data() containing department and item category mappings

    Returns:
        pd.DataFrame: Fully cleaned and enriched transaction dataset
    """
    transaction_df = clean_transaction_rd5000(external["query_0"])
    product_master_df = enrich_with_reference_data(external["query_1"], external["query_2"])
    product_master_df = standardize_dept_name(product_master_df)

    df_item_map = csv_maps["item"]
    df_dept_map = csv_maps["department"]
    product_master_df = apply_item_and_dept_mappings(product_master_df, df_item_map, df_dept_map)

    transaction_master_df = merge_transaction_with_product_details(transaction_df, product_master_df)
    final_df = clean_transaction_master_df(transaction_master_df)
    return final_df


def clean_transaction_rd5000(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans RD5000 by dropping unused columns, filling missing 'type', and formatting date/time.
    """
    df = df.drop(columns=['delivery', 'transdate', 'pos'], errors='ignore')
    df['type'] = df['type'].fillna('D')
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    df['time'] = pd.to_timedelta(df['time'], errors='coerce').apply(
        lambda td: (pd.Timestamp("00:00:00") + td).time() if pd.notnull(td) else None
    )
    return df


def enrich_with_reference_data(rd5500: pd.DataFrame, rd1800: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches RD5500 with department metadata from RD1800.
    """
    rd1800_trimmed = rd1800[['branch', 'dept_code', 'dept_name']]
    merged_df = pd.merge(
        rd5500,
        rd1800_trimmed,
        left_on=['branch', 'dep_code'],
        right_on=['branch', 'dept_code'],
        how='left'
    )
    merged_df.drop(columns=['data_id', 'unit_prc', 'pos', 'dept_code'], inplace=True, errors='ignore')
    merged_df.drop_duplicates(inplace=True)
    return merged_df


def standardize_dept_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes 'dept_name' using 'incode' prefix and business override rules.
    """
    prefix_to_dept = {
        "ADD": "ADD ONS", "BAH": "BIGGS AT HOME", "BDS": "BDS", "BEEF": "SIZZLERS", "BEF": "SIZZLERS",
        "BEV": "BEVERAGES", "BLK": "BULK ORDER", "BRK": "BREAKFAST", "BL": "BIGGS LOYALTY", "BO": "BULK ORDER",
        "BS": "BURGERS AND SANDWICHES", "BULK": "BULK ORDER", "CHM": "CHICKEN MEALS", "CKS": "CAKES",
        "CLM": "CLASSIC MEALS", "DEL": "FOOD PANDA", "EVENT": "EVENT", "F": "FUNCTION", "FND": "BURGERS AND SANDWICHES",
        "FP": "FP COFFEE", "GB": "GO BIGG", "KEF": "KIDS EAT FREE", "MI": "BIGGS MERCH ITEMS", "MOM": "MOM KNOWS BEST",
        "MP": "MARKETING PROMOS", "NES": "NESPRESSO", "PARTY": "PARTY PACKS", "PI": "PREMIUM ITEMS",
        "PP": "PARTY PACKS", "PREM": "PREMIUM ITEMS", "PST": "PASTA DISHES", "REP": "REPRESENTATION",
        "RND": "RND", "RSD": "RSD PROMOS", "SAL": "SALAD DRESSINGS", "SHK": "SHAKES", "SID": "TOPSIDERS",
        "SIZ": "SIZZLERS", "SLD": "SALAD DELIGHTS", "SUP": "SOUP", "TEST": "TEST MARKET", "UP": "UPGRADES",
        "XTR": "EXTRAS"
    }

    def extract_alpha_prefix(incode):
        return re.match(r'^[A-Z]+', str(incode)).group() if re.match(r'^[A-Z]+', str(incode)) else None

    df = df.copy()
    df['prefix'] = df['incode'].apply(extract_alpha_prefix)
    df['dept_name_std'] = df['prefix'].map(prefix_to_dept)

    df.loc[((df['dep_code'] == 53) | (df['dep_code'] == 54)) & (df['branch'] == 'BRLN'), 'dept_name_std'] = "NANAYS FAVORITE"
    df.loc[(df['dep_code'] == 42) & (df['branch'] == 'BRLN'), 'dept_name_std'] = "BIGGS LOYALTY"
    df.loc[(df['incode'] == 'SID12') & (df['branch'] == 'BRLN'), 'dept_name_std'] = "PASTA DISHES"

    df['dept_name_std'] = df['dept_name_std'].fillna(df['dept_name'])
    df.drop(columns=['dept_name', 'prefix'], inplace=True)
    return df


def apply_item_and_dept_mappings(df: pd.DataFrame, df_item_map: pd.DataFrame, df_dept_map: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the transaction data with item-level and department-level metadata.
    """
    df = df.copy()
    df_item_map.columns = df_item_map.columns.str.strip()
    df_dept_map.columns = df_dept_map.columns.str.strip()

    df_item_map = df_item_map.rename(columns={'cat_flag': 'cat_flag_item'})

    df = df.merge(
        df_item_map[['ite_desc', 'ite_desc_std', 'cat_flag_item', 'status', 'food_type']],
        on='ite_desc', how='left'
    )

    df = df.merge(
        df_dept_map[['dep_desc', 'cat_flag']],
        left_on='dept_name_std', right_on='dep_desc', how='left'
    ).rename(columns={'cat_flag': 'cat_flag_dept'})

    df['cat_flag'] = df['cat_flag_item'].combine_first(df['cat_flag_dept'])
    df.loc[df['cat_flag_dept'] == 'ignore', 'cat_flag'] = 'ignore'

    df.drop(columns=['cat_flag_item', 'cat_flag_dept', 'dep_desc'], inplace=True)
    return df


def merge_transaction_with_product_details(transaction_df: pd.DataFrame, product_master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge transaction data with enriched product master data.
    """
    df = transaction_df.copy()
    product_master_df = product_master_df.rename(columns={'incode': 'ite_code'})

    merged_df = df.merge(
        product_master_df,
        on=['branch', 'ite_code', 'dep_code'],
        how='left'
    )
    return merged_df


def clean_transaction_master_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and filters the transaction master DataFrame.
    """
    df = df.copy()
    df.drop(columns=['ite_code', 'dep_code', 'ite_desc'], inplace=True, errors='ignore')
    df = df[df['cat_flag'] != 'ignore']
    df = df[df['status'] == 'active']
    df.drop(columns=['status'], inplace=True, errors='ignore')
    return df