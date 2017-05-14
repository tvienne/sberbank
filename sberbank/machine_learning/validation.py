"""
Deals with validation in machine learning
"""
import math


def rmsle(y, pred):
    """
    WARN : ERROR in the function for the moment.
    @author : Thibaud
    Computes RMSLE validation score.
    :param y: (pandas serie) label
    :param pred: (pandas serie) prediction
    :return: rmsle
    """
    log_y1 = y.reset_index(drop=True).apply(lambda el: math.log1p(float(el)))
    log_pred1 = pred.reset_index(drop=True).apply(lambda el: math.log1p(float(el)))
    square_log_serie = (log_pred1 - log_y1).multiply((log_pred1 - log_y1))
    result = math.sqrt(square_log_serie.mean())

    return result
