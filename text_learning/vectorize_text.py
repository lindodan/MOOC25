#!/usr/bin/python3

import os
import joblib
import re
import sys
import pickle

sys.path.append(os.path.abspath("../tools/"))
from parse_out_email_text import parseOutText

from_sara = open("from_sara.txt", "r", encoding="utf-8")
from_chris = open("from_chris.txt", "r", encoding="utf-8")

from_data = []
word_data = []

temp_counter = 0

for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        temp_counter += 1
        if temp_counter < 200:  # Limit to first 200 emails for speed
            path = os.path.join('../tools/', path.strip())

            # **FIXED: Open in text mode with UTF-8 encoding**
            with open(path, "r", encoding="utf-8") as email:
                emailText = parseOutText(email)

                # Remove specific words
                for word in ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"]:
                    emailText = emailText.replace(word, "")

                word_data.append(emailText)
                from_data.append(0 if name == "sara" else 1)

print("Emails Processed")
from_sara.close()
from_chris.close()
# **FIXED: Use "wb" instead of "w" for binary pickle writing**
with open("your_word_data.pkl", "wb") as word_file:
    pickle.dump(word_data, word_file)

with open("your_email_authors.pkl", "wb") as author_file:
    pickle.dump(from_data, author_file)

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(stop_words="english")
tfidf_matrix = vect.fit_transform(word_data)

feature_names = vect.get_feature_names_out()
print(len(feature_names))  # Total number of features
