import pandas as pd
import os


def load_full_dataset():
    """
    Loads train and test dataset and concat them in order to form -> full_df
    :return: full_dataframe
    """
    # Check Path:
    if not os.path.isfile("../data/train.csv"):
        raise ValueError("ERR : not found train dataset in path ../data/train.csv")
    if not os.path.isfile("../data/test.csv"):
        raise ValueError("ERR : not found test dataset in path ../data/test.csv")
    if not os.path.isfile("../data/macro.csv"):
        raise ValueError("ERR : not found macro dataset in path ../data/macro.csv")

    # Load dataframes
    macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
                  "micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
                  "income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]

    train_df = pd.read_csv("../data/train.csv", parse_dates=['timestamp'])
    test_df = pd.read_csv("../data/test.csv", parse_dates=['timestamp'])
    macro_df = pd.read_csv("../data/macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

    # Create a features "is_test in order to easily separate the two dataframes then concat dataframes.
    train_df["is_test"] = 0
    test_df["is_test"] = 1
    test_df["price_doc"] = 0

    full_df = pd.concat([train_df, test_df])
    full_df = pd.merge_ordered(full_df, macro_df, on='timestamp', how='left')
    return full_df


def index_by_id(df):
    """
    @author : Thibaud
    indexes dataframe using column "id"
    :param df: (pandas dataframe)
    :return: dataframe + index id
    """
    return df.set_index(df["id"])
