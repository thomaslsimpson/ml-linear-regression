# Attempt Number 1: Linear Regression

import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd



def test_feature(feature_col, target_col, len_train, len_test):
	# Split the data into training/testing sets
	data_train   = feature_col[:(0 - len_test)]
	data_test    = feature_col[(0 - len_test):]
	target_train = target_col[:(0 - len_test)] 
	target_test  = target_col[(0 - len_test):] 

	# Create linear regression object
	regr = linear_model.LinearRegression()

	# Train the model using the training sets
	regr.fit(data_train, target_train)

	# Make predictions using the testing set
	prediction = regr.predict(data_test)

	# The coefficients
	print("Coefficients:", regr.coef_)
	# The mean squared error
	print("\t MSE: %.2f" % mean_squared_error(target_test, prediction))
	# The coefficient of determination: 1 is perfect prediction
	print("\t Coefficient of determination: %.2f" % r2_score(target_test, prediction))


# Load the power dataset from teh csv file
# AT,V,AP,RH,PE
# - Temperature (T) in the range 1.81°C to 37.11°C,
# - Ambient Pressure (AP) in the range 992.89-1033.30 milibar,
# - Relative Humidity (RH) in the range 25.56% to 100.16%
# - Exhaust Vacuum (V) in the range 25.36-81.56 cm Hg
# - Net hourly electrical energy output (PE) 420.26-495.76 MW (Target we are trying to predict)

df = pd.read_csv("data/CCPP_data.csv")

PERC_TRAIN = 0.7
PERC_TEST  = 0.1
PERC_VALID = 1.0 - (PERC_TRAIN + PERC_TEST)


# Pull out the target column: "PE"
hourly_output_data = df[["PE"]]
len_pe= len(hourly_output_data)
len_train = int(len_pe * PERC_TRAIN) 
len_test  = int(len_pe * PERC_VALID)
len_valid = len_pe - (len_train + len_test)


print("AT (expect high match)")
test_feature(df[["AT"]],            df[["PE"]], len_train, len_test)

print("AT, V (expecting highest match)")
test_feature(df[["AT", "V"]],       df[["PE"]], len_train, len_test)

print("AT, RH (expect same as AT)")
test_feature(df[["AT", "RH"]],       df[["PE"]], len_train, len_test)

print("AT, AP (expect same as AT)")
test_feature(df[["AT", "AP"]],      df[["PE"]], len_train, len_test)

print("AT, V, AP, RH (expect same as AT,V)")
test_feature(df[["AT", "V", "AP", "RH"]], df[["PE"]], len_train, len_test)



