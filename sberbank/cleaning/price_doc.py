"""
cleans price_doc variable
"""
import numpy as np
import pandas as pd
from sberbank.cleaning.clean_toolbox import pred_nan_values
from sberbank.features.square_meter_price import square_meter_per_area


def clean_price_doc(df):
    """
    @author : TV
    Cleans price_doc exceptions.
    WARN : make sure that your dataframe has been indexed with id col (see import_export.import_export.index_by_idea())
    WARN : make sure that life_sq and full_sq features have been cleaned.
    :param df: (pandas dataframe)
    :return: df with cleaned column "price_doc".
    """
    # In order to improve prediction, create features lsq_x1_area and fsq_x1_area:
    full_df = square_meter_per_area(df)
    # Extract train-test :
    train_df = full_df[full_df["is_test"] == 0]
    test_df = full_df[full_df["is_test"] == 1]

    # fill with nan values exceptions :
    train_df["price_doc"] = train_df["price_doc"].replace([1000000, 2000000, 3000000], np.nan)
    train_df["price_doc"] = train_df["price_doc"].apply(lambda el: np.nan if el < 1800000 else el)

    # Predict price_doc :
    features = ['fsq_x1_area', "lsq_x1_area", "floor", "max_floor", "build_year", "kitch_sq", "state", "num_room",
                "material", "metro_min_avto", "kindergarten_km", "school_km", "industrial_km", "green_zone_km",
                "railroad_km", "metro_km_avto", "park_km", "radiation_km", "area_m"]
    label = "price_doc"
    train_df = pred_nan_values(train_df, features, label, check_tuning=False)

    # reconcat the dataframes
    full_df = pd.concat([train_df, test_df])

    return full_df
