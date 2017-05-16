"""
Deals with sq features cleaning
"""
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sberbank.machine_learning.validation import rmsle,cross_val_predict,result

def clean_sq(df):
    """
    @author : thibaud
    Deals with columns full_sq and life_sq.
    WARN : make sure that your dataframe has been indexed with id column (see import_export.import_export.index_by_idea())
    :param df: (pandas dataframe)
    :return: df with processed columns "life_sq" and "full_sq".
    """
    # Drop the two sq exceptions (id = [3530, 13549])
    df = df.drop(labels=[3530, 13549], axis=0)

    # Deal with observations where life_sq > full_sq
    df["full_sq"] = df.apply(lambda row: row["life_sq"] if row["full_sq"] < row["life_sq"] else row["full_sq"],
                             axis=1)

    # Clean missing values in sq using feature mean
    #df["full_sq"] = df["full_sq"].fillna(df["full_sq"].mean())
    #df["life_sq"] = df["life_sq"].fillna(df["life_sq"].mean()) JK 16/05/2017 cf pred_nan_values

    return df


def pred_nan_values(df_, verbose=1):
    """
    @author : JK
    Prédit les NaN d'une colonne en entraînant un model RandomForestRegressor sur les autres colonnes.
    :param df: (pandas dataframe)
    :return: df with the NaN values predicted.
    """
    # columns used for training
    col_full = ['full_sq', 'floor', 'kitch_sq', 'life_sq', 'num_room', 'max_floor', 'green_zone_km',
                'kindergarten_km', 'metro_min_avto', 'workplaces_km']
    # columns used for as target ( nan prediction)
    col_order = ['num_room', 'kitch_sq', 'max_floor', 'floor', 'life_sq']

    print('--- NaN value prediction ---')

    for c in col_order:
        df = df_[col_full]

        # Train/test creation
        col_tmp = list(set(col_full) - set([c]))
        list_tmp = df[c].isnull()
        df['tmp'] = list_tmp
        x_train = df.loc[df['tmp'] == False][col_tmp]
        x_test = df.loc[df['tmp'] == True][col_tmp]
        y_train = df.loc[df['tmp'] == False][c]
        y_test = df.loc[df['tmp'] == True][c]

        # NaN values
        x_train = x_train.fillna(df.median()).values
        x_test = x_test.fillna(df.median()).values
        y_train = y_train.values

        # clf initialization
        clf = RandomForestRegressor(n_estimators=50, verbose=0, n_jobs=-1)

        # check tuning
        y_pred_rfr = cross_val_predict(x_train, y_train, clf,n_fold=3)
        mean = np.mean(y_train)
        y_pred_random_mean = np.array([mean for i in range(len(y_pred_rfr))])

        if verbose:
            print('\n' + str(c))
            result(y_train, y_pred_rfr, y_pred_random_mean)

        # final training
        clf = RandomForestRegressor(n_estimators=50, verbose=0, n_jobs=-1)
        clf.fit(x_train, y_train)
        y_test_ = clf.predict(x_test)

        # Loop for replacing the NaN values by their predicted values
        list_tmp2 = df_[c].isnull()
        df_['tmp'] = list_tmp2

        df_.ix[df_.tmp == True, c] = y_test_
        df_.drop('tmp', axis=1, inplace=True)
        df.drop('tmp', axis=1, inplace=True)
    return df_
