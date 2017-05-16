"""
Thibaud Kaggle main entry point
"""
import sys
import os
import pandas as pd
pd.options.mode.chained_assignment = None ## without boring message, I feel better
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sberbank.import_export.data_import import load_full_dataset, index_by_id
from sberbank.cleaning.categorical import yes_no_binarisation,otherise_categorical_feature,get_dummies
from sberbank.cleaning.sq import clean_sq,pred_nan_values
from sberbank.import_export.data_export import export_kaggle, export_data
from sberbank.machine_learning.split import tt_split
from sberbank.machine_learning.validation import rmsle,cross_val_predict


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
to_binarize = ["water_1line", "big_market_raion", "big_road1_1line", "culture_objects_top_25",
              "detention_facility_raion", "incineration_raion", "detention_facility_raion",
              "nuclear_reactor_raion",
              "oil_chemistry_raion", "radiation_raion", "railroad_1line", "thermal_power_plant_raion"]

# Binary categorical columns encoding :
full_df = yes_no_binarisation(full_df,to_binarize)

# Deal with sq :
full_df = clean_sq(full_df)


# Other variables
#full_df["num_room"] = full_df["num_room"].fillna(full_df["num_room"].mean())
#full_df["floor"] = full_df["floor"].fillna(full_df["floor"].mean())

full_df["kremlin_km"] = full_df["kremlin_km"].fillna(full_df["kremlin_km"].mean())


# HandCrafted
full_df["ext_sq"] = full_df["full_sq"] - full_df["life_sq"]
full_df['rel_floor'] = full_df['floor'] / full_df['max_floor'].astype(float)
full_df['rel_kitch_sq'] = full_df['kitch_sq'] / full_df['full_sq'].astype(float)

print(full_df.shape)

# Categorical & Numerical columns
col_categorical = ['timestamp','material','build_year','state','product_type',
                   'sub_area','culture_objects_top_25','thermal_power_plant_raion',
                   'incineration_raion','oil_chemistry_raion','radiation_raion',
                   'railroad_terminal_raion','big_market_raion','nuclear_reactor_raion',
                   'detention_facility_raion','ID_metro','ID_railroad_station_walk',
                   'ID_railroad_station_avto','water_1line','ID_big_road1','big_road1_1line',
                   'ID_big_road2','railroad_1line','ID_railroad_terminal','ID_bus_terminal','ecology']
#col_categorical = list(set(col_categorical) - set(to_binarize))
#full_df = get_dummies(col_categorical, full_df, 200)

#######################################
#   Features Engineering            ###
#######################################

#export_data(full_df, 'full_df')

#######################################
#   Machine learning                ###
#######################################

print("\n----- Machine learning :")

# First dirty model
# features = ["life_sq", "ext_sq", "kremlin_km", "num_room"]
features = ['full_sq', 'floor', 'kitch_sq', 'life_sq', 'num_room', 'max_floor', 'green_zone_km',
                'kindergarten_km', 'metro_min_avto', 'workplaces_km']
full_df[features] = full_df[features].fillna(full_df[features].median())
#features = full_df.columns

print('is there NaN value : ',full_df[features].isnull().values.any())

label = "price_doc"

# Train-val-test split
#full_df = full_df.fillna(full_df.mean())

train_val = full_df[full_df["is_test"] == 0]
test = full_df[full_df["is_test"] == 1]
x_train, x_val, y_train, y_val = tt_split(train_val[features], train_val[label],  0.8)
x_test = test[features]

# RandomForestRegressor
model_rfr = RandomForestRegressor(200, verbose=0,n_jobs=-1)
model_rfr.fit(x_train, y_train)
pred_train_rfr = pd.Series(model_rfr.predict(x_train))
print("Score Train : %s" % rmsle(y_train, pred_train_rfr))

model_rfr = RandomForestRegressor(200, verbose=0,n_jobs=-1)
pred_val_rfr = pd.Series(cross_val_predict(train_val[features], train_val[label], model_rfr, n_fold=5))
print("Score Val : %s" % rmsle(train_val[label], pred_val_rfr))

model_rfr = RandomForestRegressor(200, verbose=0,n_jobs=-1)

# XGB
    #TO DO : OSError: exception: access violation reading
    # https://github.com/dmlc/xgboost/issues/1163
    #  Fixed the issue by moving the path reference for MinGW-W64 to the front of the PATH system variable.
    # Cleaned, rebuilt and re-installed xgboost.
in_construction = 1
if not in_construction:
    train_val_xgb = train_val.values
    test = test.values
    df_columns = train_val.columns

    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    dtrain = xgb.DMatrix(train_val_xgb, np.log(train_val[label]), feature_names=df_columns)
    dtest = xgb.DMatrix(test, feature_names=df_columns)

    cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
                       verbose_eval=50, show_stdv=False)

    #cv_result[['train-rmse-mean', 'test-rmse-mean']].plot()
    num_boost_rounds = len(cv_result)

    #fig, ax = plt.subplots(1, 1, figsize=(8, 16))
    #xgb.plot_importance(model_xgb, max_num_features=50, height=0.5, ax=ax)

    # XGB
    model_xgb = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
    pred_test_xgb = pd.Series(model_xgb.predict(dtest))

#######################################
#   Export to Kaggle                ###
#######################################

print("\n----- Machine learning on all data / export to Kaggle :")
# RFR
model_rfr.fit(train_val[features], train_val[label])
pred_test_rfr = pd.Series(model_rfr.predict(x_test))

pred_test_rfr.index = x_test.index
test["price_doc"] = pred_test_rfr

export_kaggle(test)
