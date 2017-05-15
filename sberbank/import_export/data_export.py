"""
Deals with data exportation
"""
from time import gmtime, strftime


def export_kaggle(df, dir_path="../result/"):
    """
    @author : Thibaud
    Exports dataframe to kaggle format (columns id and price_doc). Result dataset is placed in result directory.
    WARN : dataframe has to be indexed using id column (See import_export.data_import.index_by_id())
    WARN : prediction has to be stored in column "price_doc".
    :param df: (pandas dataframe)
    :param dir_path: (str) directory where to store file.
    :return: No return
    """
    # checking
    if "price_doc" not in df.columns:
        raise ValueError("ERR : missing column price_doc. PLease make sure your prediction column name is 'price_doc'.")

    # export
    now = strftime("%Y-%m-%d_%H_%M_%S", gmtime())
    df[["price_doc"]].to_csv(dir_path+"%s.csv" % now, sep=",", index=True)

def export_data(df, name,dir_path="../saved_data/"):
    """
    @author : JK
    Exports dataframe to csv format
  
    :param df: (pandas dataframe)
    :param dir_path: (str) directory where to store file.
    :return: No return
    """

    # export
    now = strftime("%Y-%m-%d_%H_%M_%S", gmtime())
    path = dir_path+name+"%s.csv" % now
    df.to_csv(path, sep=",", index=False)
    print(name + " exported into " + path)