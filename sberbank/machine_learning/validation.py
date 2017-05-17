"""
Deals with validation in machine learning
"""
import math
import numpy as np
from sklearn.cross_validation import KFold
import pandas as pd

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


def rmse(y_true, y_pred):
    """
    @author : Thibaud JK
    Computes RMSE validation score.
    :param y: (pandas serie) label
    :param pred: (pandas serie) prediction
    :return: RMSE
    """
    square_serie = (y_pred - y_true).multiply((y_pred - y_true))
    result = math.sqrt(square_serie.mean())
    return result

def me(y_true, y_pred):
    """
    @author : Thibaud JK
    Computes ME validation score.
    :param y: (pandas serie) label
    :param pred: (pandas serie) prediction
    :return: ME
    """
    abs_serie = np.abs(y_pred - y_true)
    result = abs_serie.mean()
    return result

def cross_val_predict(x, y, clf,n_fold=5,verbose=0):
    """
    @author : JK
    Do a k-fold training to create a full predicted vector
    :param x: (pandas serie) train matrix
    :param y: (pandas serie) label
    :return: the validation predicted vector. len(y_pred) == len(x)
    """
    kf = KFold(len(y), n_folds=n_fold, shuffle=True, random_state=42)

    x = np.array(x)
    y = np.array(y)
    # Construct a kfolds object
    y_pred = np.zeros(len(y))

    temp = 1
    # Iterate through folds
    for train_index, test_index in kf:
        if verbose:
            print('N_fold : ' + str(temp) + '/' + str(n_fold))
        temp = temp + 1

        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]
        clf.fit(x_train, y_train)

        y_pred[test_index] = clf.predict(x_test)
    return pd.Series(y_pred)


def result(y, y_pred):
    """
    @author : JK
    print result ME, RMSE of a random prediction and the actual prediction
    :param y: target
    :param y_pred: predicted target
    :param y_random_mean: random target
    :return: 
    """
    ## random
    mean = y.mean()
    y_random_mean = pd.Series(np.array([mean for i in range(len(y))]))

    print('-- random_mean ')
    print('- ME : %s' % str(me(y, y_random_mean)))
    print('- RMSE : %s' % str(rmse(y, y_random_mean)))

    print('-- prediction')
    print('- ME : %s' % str(me(y, y_pred)))
    print('- RMSE : %s' % str(rmse(y, y_pred)))
    print('\n')


def log_y(y_init):
    """
    :param y_init: (pandas serie) the target
    :return: np.log(y+1) for using rmse
    """
    log_y1 = y_init.apply(lambda el: math.log1p(float(el)+1))
    return log_y1


def inv_log_y(y_final):
    """
    :param y_final: (pandas serie) the predicted target
    :return: np.exp(y_final)-1 for getting the real prediction
    """
    exp_y1 = y_final.apply(lambda el: math.exp(float(el))-2)
    return exp_y1

