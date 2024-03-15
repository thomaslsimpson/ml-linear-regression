# This is a project I did for my Duke University class on Machine Learning.

I'm leaving out some of the specifics of teh class and focusing on the data itself. While anyone taking the class could use this information to copy what I did, not using all the information will make it harder to find. That said, I suspect ChatGPT would turn up most of this anyway so I'm not all that concerned about it.

## Reviewing the data.

We start with a CSV file that contains information in several columns. We are told that the first five colums (AT, AP, V, and RH) are features and the target column, which we intend to predict as our target is "PE".

First, I reviewed the data using the `matplotlib.pyplot` python library to make some plots of the data to see what we are dealing with:

### Plots

#### Column AT plotted against PE
![AH](images/plot_feature_AT.png)

#### Column AP plotted against PE
![AP](images/plot_feature_AP.png)

#### Column V plotted against PE
![V](images/plot_feature_V.png)

#### Column RH plotted against PE
![RH](images/plot_feature_RH.png)

Colunm AT shows a very strong corelation to PE. Column V also seems to show a strong corelation. To validate this visual inspection, I looked at how closely each feature was related mathematically.

### Feature Analysis

In the attemp1 file, I just looked at one column, AT, the one that looks the strongest. See [attempt1_linear_regression.py](./attempt1_linear_regression.py).
From this, I made sure that all the python libraries are working and that nothing weird is happening. I was sure that the CSV file is loading and that the information I'm pulling is coming out right.

In Attempt 2 [attempt2_linear_regression.py](./attempt2_linear_regression.py) we are looking at each feature and trying to see how useful it might be to predict PE. We get values for each of the features:


|Feature|Coef|MSE|R2|
|-------|----|---|--|
|AT |-2.17497502|30.24|0.90|
|V  |-1.16771064|68.55|0.76|
|AP |1.49231585|213.63|0.26|
|RH |0.44896125|239.57|0.17|



