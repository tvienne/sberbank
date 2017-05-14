"""
Deals with sq features cleaning
"""


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


