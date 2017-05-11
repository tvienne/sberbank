import math
from time import gmtime, strftime


def rmsle(y, pred):
    """
    Computes RMSLE validation score.
    :param y: (pandas serie) label
    :param pred: (pandas serie) prediction
    :return: rmsle
    """
    log_y1 = math.log(y + 1)
    log_pred1 = math.log(pred + 1)
    square_log_serie = (log_y1 - log_pred1) * (log_y1 - log_pred1)
    result = math.sqrt(square_log_serie.mean())
    return result


def export_kaggle(df, dir="../result/"):
    """
    Exports dataframe to kaggle format (columns id and price_doc)
    :param df: (pandas dataframe)
    :param dir: (str) directory where to store file.
    :return: No return
    """
    if "price_doc" not in df.columns:
        raise ValueError("ERR : missing column price_doc. PLease make sure your prediction column name is 'price_doc'.")

    now = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    df[["id", "price_doc"]].to_csv(dir+"%s.csv" % now, sep=",")
