# Attempt Number 1: Linear Regression

import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd


# Load the power dataset from teh csv file
# AT,V,AP,RH,PE
# - Temperature (T) in the range 1.81°C to 37.11°C, ** wrong - (AT) is correct
# - Ambient Pressure (AP) in the range 992.89-1033.30 milibar,
# - Relative Humidity (RH) in the range 25.56% to 100.16%
# - Exhaust Vacuum (V) in the range 25.36-81.56 cm Hg
# - Net hourly electrical energy output (PE) 420.26-495.76 MW (Target we are trying to predict)

df = pd.read_csv("data/CCPP_data.csv")

# Plot outputs

plt.figure(1)
plt.title("AT")
plt.scatter(df[["AT"]], df[["PE"]], color="red")
#plt.xticks(())
#plt.yticks(())

plt.figure(2)
plt.title("AP")
plt.scatter(df[["AP"]], df[["PE"]], color="blue")
#plt.xticks(())
#plt.yticks(())

plt.figure(3)
plt.title("RH")
plt.scatter(df[["RH"]], df[["PE"]], color="green")
#plt.xticks(())
#plt.yticks(())

plt.figure(4)
plt.title("V")
plt.scatter(df[["V"]], df[["PE"]], color="purple")
#plt.xticks(())
#plt.yticks(())

plt.show()
