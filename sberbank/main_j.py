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
from sberbank.cleaning.categorical import yes_no_binarisation,otherise_categorical_feature,get_dummies,clean_cat
from sberbank.cleaning.sq import clean_sq,pred_nan_values
from sberbank.import_export.data_export import export_kaggle, export_data,import_data
from sberbank.machine_learning.split import tt_split
from sberbank.machine_learning.validation import rmse,cross_val_predict,result,log_y,inv_log_y
from sberbank.features.hand_crafted import create_features


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


#######################################
#   Importation                     ###
#######################################

saved = 0
saved_dataframe = 'full_df2017-05-18_08_07_59.csv' # Create a folder called 'saved_data'
if not saved:
    full_df = load_full_dataset()
    full_df = index_by_id(full_df)

#######################################
#   Cleaning                        ###
#######################################
# Categorical Data
    # full_df = yes_no_binarisation(full_df)
    # full_df = get_dummies(col_categorical, full_df, 200)
    full_df = clean_cat(full_df)

# Deal with sq :
    full_df = clean_sq(full_df)

#######################################
#   Features Engineering            ###
#######################################
    full_df = create_features(full_df)
    export_data(full_df, 'full_df')

else:
    full_df = import_data('full_df2017-05-18_08_07_59.csv')

#######################################
#   Machine learning                ###
#######################################

full_df = full_df.replace([np.inf, -np.inf], np.nan)
full_df = full_df.fillna(full_df.mean())
print('is there NaN value : ',full_df.isnull().values.any())

print("\n----- Machine learning :")

# First dirty model
#features = ['full_sq', 'floor', 'kitch_sq', 'life_sq', 'num_room', 'max_floor',
# 'green_zone_km','kindergarten_km', 'metro_min_avto', 'workplaces_km']#,'ext_sq','rel_floor','rel_kitch_sq']

# Getting usefull columns
label = "price_doc"
useless_col = [label] + ['id', 'id.1','is_test' ,'price_doc','timestamp']
features = list(set(full_df.columns.tolist()) - set(useless_col))

# Train-val-test split
train_val = full_df[full_df["is_test"] == 0]
test = full_df[full_df["is_test"] == 1]
train_val[label] = log_y(train_val[label])
#x_train, x_val, y_train, y_val = tt_split(train_val[features], train_val[label],  0.8)
x_test = test[features]


# RandomForestRegressor
#model_rfr = RandomForestRegressor(200, verbose=0,n_jobs=-1)
#model_rfr.fit(x_train, y_train)
#pred_train_rfr = pd.Series(model_rfr.predict(x_train))
#print("Score Train : %s" % rmse(y_train, pred_train_rfr))

model_rfr = RandomForestRegressor(20, verbose=0,n_jobs=-1)
pred_val_rfr = pd.Series(cross_val_predict(train_val[features], train_val[label], model_rfr, n_fold=5,verbose=1))
result(train_val[label],pred_val_rfr)

model_rfr = RandomForestRegressor(20, verbose=0,n_jobs=-1)

#######################################
#   Export to Kaggle                ###
#######################################

print("\n----- Machine learning on all data / export to Kaggle :")
# RFR
model_rfr.fit(train_val[features], train_val[label])
pred_test_rfr = pd.Series(model_rfr.predict(x_test))

pred_test_rfr.index = x_test.index
test["price_doc"] = inv_log_y(pred_test_rfr)

export_kaggle(test)
