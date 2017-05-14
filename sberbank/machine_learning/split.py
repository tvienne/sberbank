"""
Train_test validation split. K-fold cross validation etc...
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def tt_split(x, y, test_size=0.8):
    """
    @author : Thibaud
    Train-Test split using sklearn.
    :param x: (pandas df)
    :param y: (pandas serie)
    :param test_size: (float) train dataset proportion
    :return: x_train, x_test, y_train, y_test
    """
    if not isinstance(x, pd.DataFrame):
        raise ValueError("ERR : x has to be dataframe type.")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test

