import sys
import os
import pandas as pd
import sberbank.toolbox as tb
from sklearn.ensemble import RandomForestRegressor
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# Import data
train_df = pd.read_csv("../data/train.csv", sep=",")
test_df = pd.read_csv("../data/test.csv", sep=",")
train_df["test"] = 0
test_df["test"] = 1
test_df["price_doc"] = 0
full_df = pd.concat([train_df, test_df], axis=0)
print(full_df.shape)


# Binary categorical columns encoding :
full_df["water_1line"] = full_df["water_1line"].apply(lambda el: 1 if el == "yes" else 0)
full_df["big_market_raion"] = full_df["big_market_raion"].apply(lambda el: 1 if el == "yes" else 0)
full_df["big_road1_1line"] = full_df["big_road1_1line"].apply(lambda el: 1 if el == "yes" else 0)
full_df["culture_objects_top_25"] = full_df["culture_objects_top_25"].apply(lambda el: 1 if el == "yes" else 0)
full_df["detention_facility_raion"] = full_df["detention_facility_raion"].apply(lambda el: 1 if el == "yes" else 0)
full_df["incineration_raion"] = full_df["incineration_raion"].apply(lambda el: 1 if el == "yes" else 0)
full_df["detention_facility_raion"] = full_df["detention_facility_raion"].apply(lambda el: 1 if el == "yes" else 0)
full_df["nuclear_reactor_raion"] = full_df["nuclear_reactor_raion"].apply(lambda el: 1 if el == "yes" else 0)
full_df["oil_chemistry_raion"] = full_df["oil_chemistry_raion"].apply(lambda el: 1 if el == "yes" else 0)
full_df["radiation_raion"] = full_df["radiation_raion"].apply(lambda el: 1 if el == "yes" else 0)
full_df["railroad_1line"] = full_df["railroad_1line"].apply(lambda el: 1 if el == "yes" else 0)
full_df["thermal_power_plant_raion"] = full_df["thermal_power_plant_raion"].apply(lambda el: 1 if el == "yes" else 0)

# Clean some variable with mean
full_df["full_sq"] = full_df["full_sq"].fillna(full_df["full_sq"].mean())
full_df["life_sq"] = full_df["life_sq"].fillna(full_df["life_sq"].mean())
full_df["floor"] = full_df["floor"].fillna(full_df["floor"].mean())

# First dirty model
features = ["full_sq", "life_sq"]
label = "price_doc"

# Train-val-test split
train_val = full_df[full_df["test"] == 0]
test = full_df[full_df["test"] == 1]
x_train, x_val, y_train, y_val = tb.tt_split(train_val[features], train_val[label],  0.8)
x_test = test[features]

# Machine learning
model = RandomForestRegressor(1000, max_depth=20)
model.fit(x_train, y_train)
pred_train = pd.Series(model.predict(x_train))
pred_val = pd.Series(model.predict(x_val))
pred_test = pd.Series(model.predict(x_test))
print("Score Train : %s" % tb.rmsle(y_train, pred_train))
print("Score Val : %s" % tb.rmsle(y_val, pred_val))

# Export to Kaggle
test["price_doc"] = pred_test
tb.export_kaggle(test)
