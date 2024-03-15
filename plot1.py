# Attempt Number 1: Linear Regression
#
# Plot the four potential features compared to the target output (PE) to see which might be useful.
# Plotting the data in general will help us get some for how tot hink about the informaiton and can
# provide some insight into what we need to do.
# 

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
plt.savefig("images/plot_feature_AT.png")

plt.figure(2)
plt.title("AP")
plt.scatter(df[["AP"]], df[["PE"]], color="blue")
plt.savefig("images/plot_feature_AP.png")

plt.figure(3)
plt.title("RH")
plt.scatter(df[["RH"]], df[["PE"]], color="green")
plt.savefig("images/plot_feature_RH.png")

plt.figure(4)
plt.title("V")
plt.scatter(df[["V"]], df[["PE"]], color="purple")
plt.savefig("images/plot_feature_V.png")

plt.show()
