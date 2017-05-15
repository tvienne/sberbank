"""
Deals with categorical features
"""
import pandas as pd
from sklearn.model_selection import train_test_split

def yes_no_binarisation(df,to_binarize):
    """
    @author : Thibaud
    Binarises Yes/no variables defined in variable $to_binarize.
    :param df: (dataframe)
    :return: dataframes with binarized features
    """
    # Binary categorical columns encoding :
    #to_binarize = ["water_1line", "big_market_raion", "big_road1_1line", "culture_objects_top_25",
     #              "detention_facility_raion", "incineration_raion", "detention_facility_raion",
     #              "nuclear_reactor_raion",
     #              "oil_chemistry_raion", "radiation_raion", "railroad_1line", "thermal_power_plant_raion"]
    for col in to_binarize:
        df[col] = df[col].apply(lambda el: 1 if el == "yes" else 0)

    return df


def otherise_categorical_feature(s, treshold):
    """
    @author : JK
    Met à "others" les catégories qui apparaissent moins de treshold fois dans la série s
    :param s: (pandas serie) 
    :param treshold: le seuil d'occurences 
    :return: s_copy, (pandas serie) avec des catégories à other.

    """
    nb_lignes_par_modalite = s.value_counts()
    modalites_others = list(nb_lignes_par_modalite[nb_lignes_par_modalite < treshold].index)

    s_copy = s.copy()
    s_copy.ix[s_copy.isin(modalites_others)] = 'others'

    return s_copy


def get_dummies(col_categorical, full_df,treshold=100):
    """
    @author : JK
    Créer des colones pour les variables catégorielles.
    :param s: (pandas serie) 
    :param treshold: le seuil d'occurences 
    :return: s_copy, (pandas serie) avec des catégories à other.

    """
    print('initial full_df.shape :' , full_df.shape,'\n')
    print('--- dummification of %s columns ...' %str(len(col_categorical)))
    for var in col_categorical:
        print('- ' + str(var))
        cat_feature_otherised = otherise_categorical_feature(full_df[var],treshold)
        cat_feature_otherised_dummified = pd.get_dummies(cat_feature_otherised, prefix=var)

        full_df = pd.concat((full_df, cat_feature_otherised_dummified), axis=1)
    print('\n--- temporary_shape  :%s' %str(full_df.shape))

    # Drop categorical features
    full_df.drop(col_categorical, axis=1, inplace=True)
    print('--- droping ex-categorical columns :  %s' %str(full_df.shape))
    return full_df