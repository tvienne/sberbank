import sys
import os
import pandas as pd
import toolbox as tb
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


