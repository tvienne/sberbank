"""
Deals with Square meters price
"""
from sklearn.linear_model import LinearRegression


def square_meter_per_area(df):
    """
    @author : TV
    Combine life_sq with square meter mean price per area -> new features : lsq_per_price / lsq_x_area / lsq_x1_area
    lsq_per_area = mean life square meter price per location.
    lsq_x_area = life square price per location * life_sq.
    lsq_x1_area = affine function (life square price per location * life_sq).
    :param df: (pandas dataframe) full_df
    :return: fll_df with two new features
    """
    # Compute mean square meter price per sub_area
    known_df = df[df["is_test"] == 0]
    known_df["life_sq_price"] = df["price_doc"] / df["life_sq"]
    mean_price_per_area = known_df.groupby("sub_area").mean()["life_sq_price"].to_dict()

    # lsq_per_area and lsq_x_area
    df["lsq_per_area"] = df.apply(lambda row: mean_price_per_area[row["sub_area"]], axis=1)
    df["lsq_x_area"] = df["lsq_per_area"] * df["life_sq"]

    # lsq_x1_area (affine regression)
    known_df["lsq_per_area"] = known_df.apply(lambda row: mean_price_per_area[row["sub_area"]], axis=1)
    known_df["lsq_x_area"] = known_df["lsq_per_area"] * df["life_sq"]
    clf = LinearRegression()
    clf.fit(known_df[["lsq_x_area"]], known_df["price_doc"])
    df["lsq_x1_area"] = clf.predict(X=df[["lsq_x_area"]])

    return df
