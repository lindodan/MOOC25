#!/usr/bin/python3

import joblib
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data_unix.pkl"
authors_file = "../text_learning/your_email_authors_unix.pkl"
word_data = joblib.load( open(words_file, "rb"))
authors = joblib.load( open(authors_file, "rb") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]





### your code goes here

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
classifier = DecisionTreeClassifier()
classifier.fit(features_train, labels_train)
pred = classifier.predict(features_test)
print(accuracy_score(pred, labels_test))
feature_names = vectorizer.get_feature_names_out()
feature_importance = classifier.feature_importances_
for i,f in zip(range(len(feature_importance)),feature_importance):
    if f > 0.1:
        print(feature_names[i],f,i)