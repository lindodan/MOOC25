#!/usr/bin/python3

""" 
    Skeleton code for k-means clustering mini-project.
"""

import os
import joblib
import numpy
import matplotlib.pyplot as plt
import sys

import numpy as np

sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = joblib.load( open("../final_project/final_project_dataset_unix.pkl", "rb") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)

max = 0
min = 123123321321321
for key in data_dict.keys():
    value = data_dict[key]["salary"]
    if value == "NaN":
        continue
    value = int(value)
    if value == 0:
        continue
    if value > max:
        max = value
    if value < min:
        min = value

print("maximum: ",max)
print("minimum: ",min)
### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2,feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2, _ in finance_features:
    plt.scatter( f1, f2 )
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
kmeans = KMeans(n_clusters=3, random_state=0).fit(finance_features)
pred = kmeans.predict(finance_features)





### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("No predictions object named pred found, no clusters to plot")

scaler = MinMaxScaler()
data = np.array(data)
salary = np.array(data[:,1])
salary = salary.reshape(-1,1)
stocks = np.array(data[:,2])
stocks = stocks.reshape(-1,1)
salary = scaler.fit_transform(salary)
print("200 000",scaler.transform([[200000.]]))
stocks = scaler.fit_transform(stocks)
print("1000000",scaler.transform([[1000000.]]))