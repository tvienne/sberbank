"""
Deals with sq features cleaning
"""
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

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
    df["full_sq"] = df["full_sq"].fillna(df["full_sq"].mean())
    df["life_sq"] = df["life_sq"].fillna(df["life_sq"].mean())

    return df


def pred_nan_values(df):
    """
    @author : JK
    Prédit les NaN d'une colonne en entraînant un model RandomForestRegressor sur les autres colonnes.
    WARN : make sure that the line x_train = x_train.fillna(0) doesn't change the variable df ! 
    TO DO : Verify this function
    :param df: (pandas dataframe)
    :return: df with the NaN values predicted.
    """

    ### Get the column having NaNs values.
    nuls = []
    for c in df.columns:
        nuls.append(df[c].isnull().sum())

    pct = [round((n * 100) / len(df), 0) for n in nuls]
    df_nuls = pd.DataFrame({'col': df.columns, 'nan_count': nuls, '%': pct})
    df_nuls = df_nuls.sort_values('nan_count', ascending=True)

    col_full = df.columns

    nan_count = df_nuls['nan_count'].tolist()
    cols = df_nuls['col'].tolist()

    print('--- NaN value prediction ---')
    ### Do the trainings by beginning by the columns having the fewest amount of NaNs.
    for i in range(len(df_nuls)):
        if nan_count[i] != 0:
            print(str(cols[i]) + ', ' + str(i) + '/' + str(len(nuls))+ ', '+ str(nan_count[i]) + ' Nan Values---')

            ### clf initialization
            clf_tmp = RandomForestRegressor(n_estimators=20, verbose=1, n_jobs=-1)

            ### Train/test creation
            col_tmp = list(set(col_full) - set([cols[i]]))
            list_tmp = df[cols[i]].isnull()
            df['tmp'] = list_tmp
            x_train = df[df['tmp'] == False][col_tmp]
            x_test = df[df['tmp'] == True][col_tmp]
            y_train = df[df['tmp'] == False][cols[i]]
            y_test = df[df['tmp'] == True][cols[i]]

            df.drop('tmp', axis=1, inplace=True)

            ### NaN values
            x_train = x_train.fillna(0)
            x_test = x_test.fillna(0)

            clf_tmp.fit(x_train, y_train)
            y_test_ = clf_tmp.predict(x_test)

            ### Loop for replacing the NaN values by their predicted values
            k = df[cols[i]].tolist()
            k_tmp = list_tmp.tolist()
            compteur = 0
            for j in range(len(k_tmp)):
                if k_tmp[j] == True:
                    k[j] = y_test_[compteur]
                    compteur = compteur + 1
                    df[cols[i]] = k
    return df,nuls
