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


Next, I decided to look at performance of the variables in different combinations. Since the AP and RH column seem to ahve such little impact, I was curious about whether I needed to oncude them or not. It seems to me that including feautes that do not add value is a great way to end up with overfitting.

I moved on to Attempt 4 [attempt4_linear_regression.py](./attempt4_linear_regression.py) which I used to demonstrate that in the end, all four features were adding something to the prediction.


### 80/20 Training tests
|Features                        	|MSE	|R2	|MAPE|
|['AT']                          	|28.912109791641537	|0.9003229981796107	|0.9461589669518153|
|['V']                           	|69.97944860731239	|0.7587397918557475	|1.4236029761403044|
|['AP']                          	|212.53353291177342	|0.26727224337446687	|2.6383549474589487|
|['RH']                          	|246.02522306341206	|0.15180674174656716	|2.89850355054695|
|['AT', 'V']                     	|24.068386312654486	|0.9170221542602933	|0.858466001800614|
|['AT', 'AP']                    	|28.418942151546883	|0.9020232363190522	|0.9450113689829289|
|['AT', 'RH']                    	|22.653867323134552	|0.9218988309507683	|0.8364724452436425|
|['AT', 'V', 'AP', 'RH']         	|20.273705999687444	|0.9301046431962188	|0.7932984848767791|

### Cross Validation
|Features	|Scores	|Mean|
|['AT']                          	|[0.8994794  0.89733768 0.90302574 0.89915667 0.89512129]	|0.8988241562050259|
|['V']                           	|[0.75130658 0.75093286 0.76133905 0.75569749 0.76234696]	|0.7563245853641749|
|['AP']                          	|[0.24548431 0.29514638 0.27909191 0.25911605 0.25980214]	|0.2677281591755981|
|['RH']                          	|[0.14979816 0.13490394 0.14069268 0.16001713 0.17082364]	|0.15124710888877207|
|['AT', 'V']                     	|[0.91639124 0.91209361 0.92058171 0.91573174 0.91332637]	|0.9156249330622114|
|['AT', 'AP']                    	|[0.90125179 0.90034534 0.90373581 0.90115853 0.89677377]	|0.9006530489382859|
|['AT', 'RH']                    	|[0.92363973 0.91990374 0.92703639 0.91829337 0.91516422]	|0.9208074888487869|
|['AT', 'V', 'AP', 'RH']         	|[0.93053597 0.92681472 0.93389127 0.92680208 0.92464499]	|0.9285378066739307|


## Conclusions

The final table demonstrates pretty clearly that all four features give the best results. 


