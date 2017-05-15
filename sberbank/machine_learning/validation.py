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


"""
Do a k-fold training to create a full predicted vector
Procede à un entrainement k-fold pour recrer un vecteur complet de prédiction
"""


def cross_val_predict(x, y, clf,n_fold=5):

    kf = KFold(len(y), n_folds=n_fold, shuffle=True, random_state=42)

    x = np.array(x)
    y = np.array(y)
    # Construct a kfolds object
    y_prob = np.zeros(len(y))

    temp = 1
    # Iterate through folds
    for train_index, test_index in kf:
        print('N_fold : ' + str(temp) + '/' + str(n_fold))
        temp = temp + 1

        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]
        clf.fit(x_train, y_train)

        y_prob[test_index] = clf.predict(x_test)
    return y_prob
