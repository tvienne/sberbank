"""
Thibaud Kaggle main entry point
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sberbank.import_export.data_import import index_by_id
from sklearn import preprocessing
from sberbank.import_export.data_export import export_kaggle
from sberbank.machine_learning.split import tt_split
from sberbank.machine_learning.validation import rmsle
from sklearn.ensemble import GradientBoostingRegressor
pd.options.mode.chained_assignment = None

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

#######################################
#   Import data                     ###
#######################################

full_df = pd.read_csv("../data/ml_df.csv", parse_dates=['timestamp'])
full_df = index_by_id(full_df)

#######################################
#   Machine learning                ###
#######################################

print("\n----- Machine learning :")

for f in full_df.columns:
    if full_df[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(full_df[f].values))
        full_df[f] = lbl.transform(list(full_df[f].values))

features = ['fsq_x1_area', "lsq_x1_area", "floor", "max_floor", "build_year", "kitch_sq", "state", "num_room",
            "material", "metro_min_avto", "kindergarten_km", "school_km", "industrial_km", "sub_area",
            "green_zone_km", "railroad_km", "metro_km_avto", "park_km", "radiation_km", "area_m"]

features = [feat for feat in full_df.columns if feat not in ["id", "price_doc", "is_test", "timestamp"]]
label = "price_doc"
full_df[features] = full_df[features].fillna(full_df[features].mean())

# Train-val-test split
train_val = full_df[full_df["is_test"] == 0]
test = full_df[full_df["is_test"] == 1]
x_train, x_val, y_train, y_val = tt_split(train_val[features], train_val[label],  0.8)
x_test = test[features]

# RandomForestRegressor
params_xgb = {'n_estimators': 250, 'max_depth': 5, 'subsample': 1, 'learning_rate': 0.05, 'random_state': 42}
model = GradientBoostingRegressor(**params_xgb)
model.fit(x_train, np.log(y_train))
pred_train = pd.Series(np.exp(model.predict(x_train)))
pred_val = pd.Series(np.exp(model.predict(x_val)))
print("Score Train : %s" % rmsle(y_train, pred_train))
print("Score Val : %s" % rmsle(y_val, pred_val))

#######################################
#   Export to Kaggle                ###
#######################################

print("\n----- Machine learning on all data / export to Kaggle :")
model.fit(train_val[features], np.log(train_val[label]))
pred_test = pd.Series(np.exp(model.predict(x_test)))

pred_test.index = x_test.index
test["price_doc"] = pred_test

export_kaggle(test)
