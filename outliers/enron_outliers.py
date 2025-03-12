#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit
import pandas as pd


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset_unix.pkl", "rb") )
features = ["salary", "bonus"]
data_dict.pop("TOTAL",0)
data = featureFormat(data_dict, features)
print(data)
max_sal = 0
i = 0
for index, data_format in enumerate(data):
    salary = data_format[0]
    bonus = data_format[1]
    if salary > max_sal:
        max_sal = salary
        i = index

    ### your code below
    plt.scatter(salary, bonus)
plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.show()
print(i)
print(data_dict)
print(max_sal)