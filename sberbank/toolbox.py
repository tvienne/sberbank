import math
from sklearn.model_selection import train_test_split
from time import gmtime, strftime
import pandas as pd


def rmsle(y, pred):
    """
    Computes RMSLE validation score.
    :param y: (pandas serie) label
    :param pred: (pandas serie) prediction
    :return: rmsle
    """
    log_y1 = y.apply(lambda el: math.log(el + 1)).reset_index(drop=True)
    log_pred1 = pred.apply(lambda el: math.log(el + 1))
    square_log_serie = (log_y1 - log_pred1) ** 2
    result = math.sqrt(square_log_serie.mean())
    return result


def export_kaggle(df, dir_path="../result/"):
    """
    Exports dataframe to kaggle format (columns id and price_doc)
    :param df: (pandas dataframe)
    :param dir_path: (str) directory where to store file.
    :return: No return
    """
    if "price_doc" not in df.columns:
        raise ValueError("ERR : missing column price_doc. PLease make sure your prediction column name is 'price_doc'.")

    now = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    df[["id", "price_doc"]].to_csv(dir_path+"%s.csv" % now, sep=",", index=False)


def tt_split(x, y, test_size=0.8):
    """
    Train-Test split using sklearn
    :param x: (pandas df)
    :param y: (pandas serie)
    :param test_size:
    :return:
    """
    if not isinstance(x, pd.DataFrame):
        raise ValueError("ERR : x has to be dataframe type.")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test
