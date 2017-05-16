"""
Deals with validation in machine learning
"""
import math
import numpy as np
from sklearn.cross_validation import KFold


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

def rmse(y_true,y_pred):
    """
    Root Mean Squared Error
    @author : JK
    :param y_true: labels
    :param y_pred: predicted labels
    :return: Root Mean Squared Error
    """

    k = 0
    N = len(y_true)
    for i in range(N):
        k = k + (y_pred[i]-y_true[i])**2
    k = np.sqrt(k/N)
    return k

def me(y_true,y_pred):
    """
    Mean Error
    @author : JK
    :param y_true: labels
    :param y_pred: predicted labels
    :return: Mean Error
    """
    k = 0
    N = len(y_true)
    for i in range(N):
        k = k + abs(y_pred[i]-y_true[i])
    k = k/N
    return k

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
    return y_pred


def result(y, y_pred, y_random_mean):
    """
    @author : JK
    print result ME, RMSE of a random prediction and the actual prediction
    :param y: target
    :param y_pred: predicted target
    :param y_random_mean: random target
    :return: 
    """
    ## random
    print('\n-- random_mean ')
    print('- ME : %s' % str(me(y, y_random_mean)))
    print('- RMSE : %s' % str(rmse(y, y_random_mean)))

    print('-- prediction')
    print('- ME : %s' % str(me(y, y_pred)))
    print('- RMSE : %s' % str(rmse(y, y_pred)))