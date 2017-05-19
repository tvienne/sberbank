"""
Thibaud Kaggle main entry point
"""
import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sberbank.import_export.data_import import load_full_dataset, index_by_id
from sberbank.cleaning.categorical import yes_no_binarisation
from sberbank.cleaning.sq import clean_sq
from sberbank.import_export.data_export import export_kaggle
from sberbank.machine_learning.split import tt_split
from sberbank.machine_learning.validation import rmsle
from sberbank.features.square_meter_price import square_meter_per_area
pd.options.mode.chained_assignment = None


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

# Deal with life and full sq:
full_df = clean_sq(full_df)

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
#   Machine learning                ###
#######################################

print("\n----- Machine learning :")

# First dirty model
features = ['lsq_x_area', "ext_sq", 'num_room', "kremlin_km", "floor", "state"]
label = "price_doc"
full_df[features] = full_df[features].fillna(full_df[features].mean())

# Train-val-test split
train_val = full_df[full_df["is_test"] == 0]
test = full_df[full_df["is_test"] == 1]
x_train, x_val, y_train, y_val = tt_split(train_val[features], train_val[label],  0.8)
x_test = test[features]

# RandomForestRegressor
model = RandomForestRegressor(1000, max_depth=7)
model.fit(x_train, y_train)
pred_train = pd.Series(model.predict(x_train))
pred_val = pd.Series(model.predict(x_val))
print("Score Train : %s" % rmsle(y_train, pred_train))
print("Score Val : %s" % rmsle(train_val[label], pred_val))

#######################################
#   Export to Kaggle                ###
#######################################

print("\n----- Machine learning on all data / export to Kaggle :")
model.fit(train_val[features], train_val[label])
pred_test = pd.Series(model.predict(x_test))

pred_test.index = x_test.index
test["price_doc"] = pred_test

export_kaggle(test)
