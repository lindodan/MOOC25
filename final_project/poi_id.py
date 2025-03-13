#!/usr/bin/python

import sys
import pickle
import os
sys.path.append(os.path.abspath(("../tools/")))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,recall_score,precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
    "exercised_stock_options",
    "total_stock_value",
    "bonus",
    "salary",
    "long_term_incentive",
    "restricted_stock",
    "deferred_income"]

### Load the dictionary containing the dataset
with open("../final_project/final_project_dataset_unix.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL",0)
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

pipeline = ImbPipeline([
    ("smote", SMOTE(sampling_strategy=0.7, random_state=42)),
    ("scaler", StandardScaler()),  # preprocess data
    ("svm", SVC(probability=False, random_state=42))  # SVM model
])

# Define parameter grid for GridSearchCV
param_grid_svm = {
    "svm__C": [0.1, 1, 10],
    "svm__kernel": ["rbf", "linear"],
    "svm__gamma": ["scale", 0.01, 0.1, 1],
}

# Set up GridSearchCV with recall scoring
grid_search_svm = GridSearchCV(pipeline, param_grid_svm, scoring="recall", cv=5, n_jobs=-1)

# Train model using GridSearchCV (pipeline ensures correct preprocessing)
grid_search_svm.fit(features_train, labels_train)

# Get the best model
best_svm = grid_search_svm.best_estimator_

# Predict on test data
predictions_svm = best_svm.predict(features_test)

# Print best parameters and classification report
print("Best Parameters for SVM:", grid_search_svm.best_params_)
print("Classification Report (SVM):")
print(classification_report(labels_test, predictions_svm))

clf = best_svm

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
