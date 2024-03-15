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



