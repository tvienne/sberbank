"""
Deals with sq features cleaning
"""
import numpy as np
from sberbank.cleaning.clean_toolbox import pred_nan_values


def clean_sq(df):
    """
    @author : TV
    Deals with columns full_sq and life_sq exceptions
    WARN : make sure that your dataframe has been indexed with id col (see import_export.import_export.index_by_idea())
    :param df: (pandas dataframe)
    :return: df with processed columns "life_sq" and "full_sq".
    """
    # Drop the two sq exceptions (id = [3530, 13549])
    df = df.drop(labels=[3530, 13549], axis=0)

    # Deal with too small values in life_sq or full_sq :
    df["life_sq"] = df["life_sq"].apply(lambda el: np.nan if el < 5.0 else el)
    df["full_sq"] = df["full_sq"].apply(lambda el: np.nan if el < 8.0 else el)

    # Trying to fill full_sq NANs based on life_sq. If not possible, infer from full_sq mean.
    df["full_sq"] = df.apply(lambda row: row["life_sq"] if row["full_sq"] == np.nan else row["full_sq"], axis=1)
    df["full_sq"] = df["full_sq"].fillna(df["full_sq"].mean())

    # if life_sq > full_sq, lower life_sq to full_sq value.
    df["life_sq"] = df.apply(lambda row: apply_life_full_exception(row["life_sq"], row["full_sq"]), axis=1)

    # Predict missing life_sq using pred_nan_values function
    features = ['full_sq', 'floor', 'kitch_sq', 'num_room']
    label = "life_sq"
    df = pred_nan_values(df, features, label, False)

    return df


def apply_life_full_exception(life, full):
    """
    @author: TV
    Deals with observations where life_sq > full_sq.
    WARN : Make sure there are not any NaN in full_sq, otherwise function will raise an error.
    :param life: (float) life_sq
    :param full:  (float) full_sq
    :return: possibly new life_sq
    """
    if life == np.nan:  # case life==nan
        return np.nan
    else:
        if full == np.nan:
            raise ValueError("ERR : full_sq equals np.nan. please make sure there are no NaN values in full_sq")
        if life > full:
            return full
        else:
            return life
