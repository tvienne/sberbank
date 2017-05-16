"""
Deals with sq features cleaning
"""
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sberbank.machine_learning.validation import cross_val_predict
import matplotlib.pyplot as plt
import pandas as pd


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
    features = ['full_sq', 'floor', 'kitch_sq', 'num_room', 'max_floor', 'green_zone_km',
                'kindergarten_km', 'metro_min_avto', 'workplaces_km']
    label = "life_sq"
    df = pred_nan_values(df, features, label, True)
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


def pred_nan_values(df, features, label, verbose=True):
    """
    @author : JK, TV
    Prédit les NaN d'une colonne en entraînant un model RandomForestRegressor sur les colonnes features.
    :param df: (pandas dataframe)
    :param features: (list) containing features names
    :param label: (str) label column name
    :param verbose: (bool) whether or not display graphics.
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

    # clf initialization
    clf = RandomForestRegressor(n_estimators=50, verbose=0, n_jobs=-1)

    # check tuning
    y_pred_rfr = cross_val_predict(x_train, y_train, clf, n_fold=3)
    # mean = np.mean(y_train)
    # y_pred_random_mean = np.array([mean for i in range(len(y_pred_rfr))])

    if verbose:
        # result(y_train, y_pred_rfr, y_pred_random_mean)
        plt.scatter(y_train, y_pred_rfr)
        plt.plot(np.linspace(0, 800, 100), np.linspace(0, 800, 100), color="r")
        plt.plot(np.linspace(0, 800, 100),  2.0 * np.linspace(0, 800, 100), color="g")
        plt.plot(np.linspace(0, 800, 100), 0.5 * np.linspace(0, 800, 100), color="g")
        plt.xlabel("%s" % label)
        plt.ylabel("prediction")
        plt.title("Prediction on %s evaluation : (green scale = x2.0)" % label)
        plt.show()

    # final training and fill na
    clf = RandomForestRegressor(n_estimators=50, verbose=0, n_jobs=-1)
    clf.fit(x_train, y_train)
    y_test = pd.Series(clf.predict(x_test), x_test.index)
    df[label] = df[label].fillna(y_test)
    print(df[label])

    return df
