"""
Deals with categorical features
"""


def yes_no_binarisation(df):
    """
    @author : Thibaud
    Binarises Yes/no variables defined in variable $to_binarize.
    :param df: (dataframe)
    :return: dataframes with binarized features
    """
    # Binary categorical columns encoding :
    to_binarize = ["water_1line", "big_market_raion", "big_road1_1line", "culture_objects_top_25",
                   "detention_facility_raion", "incineration_raion", "detention_facility_raion",
                   "nuclear_reactor_raion",
                   "oil_chemistry_raion", "radiation_raion", "railroad_1line", "thermal_power_plant_raion"]
    for col in to_binarize:
        df[col] = df[col].apply(lambda el: 1 if el == "yes" else 0)

    return df

