"""
Cleaning tools
"""
from sberbank.machine_learning.validation import cross_val_predict, result
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def pred_nan_values(df, features, label, check_tuning=False):
    """
    @author : JK, TV
    Prédit les NaN d'une colonne en entraînant un model RandomForestRegressor sur les colonnes features.
    :param df: (pandas dataframe)
    :param features: (list) containing features names
    :param label: (str) label column name
    :param check_tuning: (bool) whether or not display graphics and results.
    :return: df with the NaN values predicted.
    """
    print("--- NaN value prediction on column : %s ---" % label)

    # Train/test creation
    x_train = df[df[label].notnull()][features]
    x_test = df[df[label].isnull()][features]
    y_train = df[df[label].notnull()][label]

    # NaN values
    x_train = x_train.fillna(df.median())
    x_test = x_test.fillna(df.median())

    if check_tuning:
        # check tuning
        clf = RandomForestRegressor(n_estimators=150, verbose=0, n_jobs=-1)
        y_pred_rfr = cross_val_predict(clf, x_train, y_train, n_fold=3)
        result(y_train.values, y_pred_rfr)

        # Plot result
        plt.scatter(y_train, y_pred_rfr)
        plt.plot(np.linspace(0, 800, 100), np.linspace(0, 800, 100), color="r")
        plt.plot(np.linspace(0, 800, 100),  2.0 * np.linspace(0, 800, 100), color="g")
        plt.plot(np.linspace(0, 800, 100), 0.5 * np.linspace(0, 800, 100), color="g")
        plt.xlabel("%s" % label)
        plt.ylabel("prediction")
        plt.title("Prediction on %s evaluation : (green scale = x2.0)" % label)
        plt.show()

    # final training and fill na
    clf = RandomForestRegressor(n_estimators=150, verbose=0, n_jobs=-1)
    clf.fit(x_train, y_train)
    y_test = pd.Series(clf.predict(x_test), x_test.index)
    df[label] = df[label].fillna(y_test)

    return df
