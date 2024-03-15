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

	MSE= mean_squared_error(target_test, prediction)
	R2 = r2_score(target_test, prediction)
	print(f"{regr.coef_}\t{MSE:.2f}\t{R2:.2f}")

	# The coefficients
	#print("Coefficients: ", regr.coef_)
	# The mean squared error
	#print("MSE: %.2f" % mean_squared_error(target_test, prediction))
	# The coefficient of determination: 1 is perfect prediction
	#print("R2: %.2f" % r2_score(target_test, prediction))

	# Plot outputs
	plt.scatter(data_test, target_test, color="black")
	plt.plot(data_test, prediction, color="blue", linewidth=3)
	plt.xticks(())
	plt.yticks(())
	plt.show()


# Load the power dataset from teh csv file
# AT,V,AP,RH,PE
# - Temperature (T) in the range 1.81°C to 37.11°C,
# - Ambient Pressure (AP) in the range 992.89-1033.30 milibar,
# - Relative Humidity (RH) in the range 25.56% to 100.16%
# - Exhaust Vacuum (V) in the range 25.36-81.56 cm Hg
# - Net hourly electrical energy output (PE) 420.26-495.76 MW (Target we are trying to predict)

df = pd.read_csv("data/CCPP_data.csv")

# Pull out the target column: "PE"
hourly_output_data = df[["PE"]]
len_pe= len(hourly_output_data)
len_train = int(len_pe * 0.8)
len_test = len_pe - len_train

print("AT", end="\t")
test_feature(df[["AT"]], df[["PE"]], len_train, len_test)

print("V", end="\t")
test_feature(df[["V"]], df[["PE"]], len_train, len_test)

print("AP", end="\t")
test_feature(df[["AP"]], df[["PE"]], len_train, len_test)

print("RH", end="\t")
test_feature(df[["RH"]], df[["PE"]], len_train, len_test)

