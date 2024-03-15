# Attempt Number 1: Linear Regression

import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd

# Load the power dataset from teh csv file
# AT,V,AP,RH,PE
# - Temperature (T) in the range 1.81°C to 37.11°C,
# - Ambient Pressure (AP) in the range 992.89-1033.30 milibar,
# - Relative Humidity (RH) in the range 25.56% to 100.16%
# - Exhaust Vacuum (V) in the range 25.36-81.56 cm Hg
# - Net hourly electrical energy output (PE) 420.26-495.76 MW (Target we are trying to predict)

df = pd.read_csv("data/CCPP_data.csv")

# Pull out just the Temperature (and preserve the Dataframe shape by using double brackets)
temperature_data = df[["AT"]]

len_temperature= len(temperature_data)
len_train = int(len_temperature * 0.8)
len_test = len_temperature - len_train

print(f"Total[{len_temperature}] train[{len_train}] test[{len_test}]")

# Pull out the target PE
hourly_output_data = df[["PE"]]
print(temperature_data)
print(hourly_output_data)


# Split the data into training/testing sets
temperature_data_train = temperature_data[:(0 - len_test)]
temperature_data_test  = temperature_data[(0 - len_test):]

target_train = hourly_output_data[:(0 - len_test)] 
target_test  = hourly_output_data[(0 - len_test):] 


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
#regr.fit(diabetes_X_train, diabetes_y_train)
regr.fit(temperature_data_train, target_train)

# Make predictions using the testing set
#diabetes_y_pred = regr.predict(diabetes_X_test)
prediction = regr.predict(temperature_data_test)


# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(target_test, prediction))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(target_test, prediction))



# Plot outputs
#plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.scatter(temperature_data_test, target_test, color="black")

#plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)
plt.plot(temperature_data_test, prediction, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

