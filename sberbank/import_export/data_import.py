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

    # Load dataframes
    train_df = pd.read_csv("../data/train.csv", sep=",")
    test_df = pd.read_csv("../data/test.csv", sep=",")

    # Create a features "is_test in order to easily separate the two dataframes then concat dataframes.
    train_df["is_test"] = 0
    test_df["is_test"] = 1
    test_df["price_doc"] = 0
    full_df = pd.concat([train_df, test_df], axis=0)
    return full_df


def index_by_id(df):
    """
    @author : Thibaud
    indexes dataframe using column "id"
    :param df: (pandas dataframe)
    :return: dataframe + index id
    """
    return df.set_index(df["id"])
