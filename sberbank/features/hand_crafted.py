"""
Deals with the creation of new columns 
"""


def create_features(df):
    """
    @author : JK
    :param df: ( Pandas Dataframe) of the competition with cat(train,test,macro)
    :return: ( Pandas Dataframe) 
    """

    # Add the exterior surface
    df["ext_sq"] = df["full_sq"] - df["life_sq"]

    # Add the relative floor position of the flat
    df['rel_floor'] = df['floor'] / df['max_floor'].astype(float)

    # Add the propotion of the kitchen
    df['rel_kitch_sq'] = df['kitch_sq'] / df['full_sq'].astype(float)

    # Add month-year
    month_year = (df.timestamp.dt.month + df.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    df['month_year_cnt'] = month_year.map(month_year_cnt_map)

    # Add week-year count
    week_year = (df.timestamp.dt.weekofyear + df.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    df['week_year_cnt'] = week_year.map(week_year_cnt_map)

    # Add month and day-of-week
    df['month'] = df.timestamp.dt.month
    df['dow'] = df.timestamp.dt.dayofweek

    return df