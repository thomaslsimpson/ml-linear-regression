#
# Format of the data we are loading:
#
# AT,V,AP,RH,PE
# - Temperature (T) in the range 1.81°C to 37.11°C,
# - Ambient Pressure (AP) in the range 992.89-1033.30 milibar,
# - Relative Humidity (RH) in the range 25.56% to 100.16%
# - Exhaust Vacuum (V) in the range 25.36-81.56 cm Hg
# - Net hourly electrical energy output (PE) 420.26-495.76 MW (Target we are trying to predict)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

CPAD=32

def mape(test_col, pred_col):
	return np.mean(np.abs((test_col - pred_col) / test_col)) * 100


def runFit(label, X, y):
	X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state= 42)
	model= LinearRegression()
	model.fit(X_train, y_train)
	test_predictions= model.predict(X_test)
	mse= mean_squared_error(y_test, test_predictions)
	mape_test= mape(y_test, test_predictions)
	r2_test = r2_score(y_test, test_predictions)
	print(f'|{str(label):<{CPAD}}\t|{mse}\t|{r2_test}\t|{mape_test}|')
	

def runCV(label, X, y):
	model = LinearRegression()
	scores = cross_val_score(model, X, y, cv=5)
	print(f'|{str(label):<{CPAD}}\t|{scores}\t|{scores.mean()}|')



# Load the CSV file
df = pd.read_csv('data/CCPP_data.csv')


tests= [
	['AT'],
	['V'],
	['AP'],
	['RH'],
	['AT', 'V'],
	['AT', 'AP'],
	['AT', 'RH'],
	['AT', 'V', 'AP', 'RH'],
]

# Run tests
print("80/20 Training tests")
print(f'|{"Features":<{CPAD}}\t|MSE\t|R2\t|MAPE|')
for COLUMNS in tests:
	runFit(COLUMNS, df[COLUMNS], df['PE'])


print("Cross Validation")
print("|Features\t|Scores\t|Mean|")
for COLUMNS in tests:
	runCV(COLUMNS, df[COLUMNS], df['PE'])


