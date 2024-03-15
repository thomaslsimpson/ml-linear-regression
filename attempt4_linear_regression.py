# AT,V,AP,RH,PE
# - Temperature (T) in the range 1.81°C to 37.11°C,
# - Ambient Pressure (AP) in the range 992.89-1033.30 milibar,
# - Relative Humidity (RH) in the range 25.56% to 100.16%
# - Exhaust Vacuum (V) in the range 25.36-81.56 cm Hg
# - Net hourly electrical energy output (PE) 420.26-495.76 MW (Target we are trying to predict)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Load the CSV file
df = pd.read_csv('data/CCPP_data.csv')


def mape(test_col, pred_col):
	return np.mean(np.abs((test_col - pred_col) / test_col)) * 100



def runFit(X, y):
	# Split the data into training+validation and testing sets
	X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Split the training+validation set into actual training and validation sets
	X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

	# Initialize the linear regression model
	model = LinearRegression()

	# Fit the model on the training data
	model.fit(X_train, y_train)

	# Make predictions on the validation and test data
	predictions_val = model.predict(X_val)
	predictions_test = model.predict(X_test)

	# Calculate MSE for the validation and test sets
	mse_val = mean_squared_error(y_val, predictions_val)
	mse_test = mean_squared_error(y_test, predictions_test)
	mape_test= mape(y_test, predictions_test)
	r2_test = r2_score(y_test, predictions_test)

	# Calculate MSE
	print(f"MSE Validation: {mse_val} \tTest: {mse_test} \tR2: {r2_test} \tMAPE: {mape_test}")




# Select the first three columns as features and the fourth as the target
runFit(df[['AT']], df['PE'])
runFit(df[['V']], df['PE'])
runFit(df[['AT', 'V', 'RH']], df['PE'])
runFit(df[['AT', 'V', 'AP', 'RH']], df['PE'])

