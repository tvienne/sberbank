"""
Thibaud Dataset Processing Main entry point
"""
import os
import sys
from sberbank.import_export.data_import import load_full_dataset, index_by_id
from sberbank.cleaning.categorical import yes_no_binarisation
from sberbank.cleaning.sq import clean_sq
from sberbank.cleaning.price_doc import clean_price_doc
from sberbank.features.square_meter_price import square_meter_per_area
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


#######################################
#   Importation                     ###
#######################################

full_df = load_full_dataset()
full_df = index_by_id(full_df)
print(full_df.shape)

#######################################
#   Cleaning                        ###
#######################################

# Binary categorical columns encoding :
full_df = yes_no_binarisation(full_df)

# Deal with life and full sq and price_doc
full_df = clean_sq(full_df)
full_df = clean_price_doc(full_df)

# Other variables
full_df["num_room"] = full_df["num_room"].fillna(full_df["num_room"].mean())
full_df["floor"] = full_df["floor"].fillna(full_df["floor"].mean())
full_df["kremlin_km"] = full_df["kremlin_km"].fillna(full_df["kremlin_km"].mean())


#######################################
#   Features Engineering            ###
#######################################

full_df["ext_sq"] = full_df["full_sq"] - full_df["life_sq"]
full_df = square_meter_per_area(full_df)


#######################################
#   Export data                     ###
#######################################

full_df.to_csv("../data/ml_df.csv", sep=",")
