#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""
import os
import joblib
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,recall_score,precision_score
from sklearn.model_selection import train_test_split
import numpy as np


sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

data_dict = joblib.load(open("../final_project/final_project_dataset_unix.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



clf = DecisionTreeClassifier()
clf = clf.fit(features, labels)
pred = clf.predict(features)
#print accuracy
print(accuracy_score(pred,labels))

### it's all yours from here forward!

# make train test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(accuracy_score(pred,y_test))

# How many of POI are in the test set
print(np.sum(y_test))
print(y_test)
print(pred)
# How many people total are in the test set
print(len(y_test))

# ALL THE METRICS
print("PRECISION SCORE: ", precision_score(y_test, pred))
print("RECALL SCORE: ", recall_score(y_test, pred))
print("CONFUSION MATRIX: ", confusion_matrix(y_test, pred))
print("CLASSIFICATION REPORT: ", classification_report(y_test, pred))