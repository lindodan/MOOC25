#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib
import pandas as pd

enron_data = joblib.load(open("../final_project/final_project_dataset_unix.pkl", "rb"))
with open("../final_project/poi_names.txt") as poi_names_file:
    lines = poi_names_file.read().splitlines()
    poi_names = lines[2:] # skip first two lines

df = pd.DataFrame.from_dict(enron_data, orient='index')
print(df.columns)

number_of_people = df.shape[0]
print("Number of people in database: ", number_of_people)
number_of_features = df.shape[1]
print("Number of features: ", number_of_features)
number_of_poi = df["poi"].sum()
print("Number of POI: ", number_of_poi)
number_of_total_poi = len(poi_names)
print("Number of total POI: ", number_of_total_poi)

print("PRENTICE JAMES stock value: ",df.loc["PRENTICE JAMES"]['total_stock_value'])
print("COLWELL WESLEY number of from this person to poi emails: ",df.loc["COLWELL WESLEY"]['from_this_person_to_poi'])
print("SKILLING JEFFREY K exercised stock options: ", df.loc["SKILLING JEFFREY K"]["exercised_stock_options"])
print("SKILLING JEFFREY K total_payments: ", df.loc["SKILLING JEFFREY K"]["total_payments"])
print("LAY KENNETH L total_payments: ", df.loc["LAY KENNETH L"]["total_payments"])
print("FASTOW ANDREW S total_payments: ", df.loc["FASTOW ANDREW S"]["total_payments"])

quantified_salary_number = len(df[df["salary"] != "NaN"])
print("Number of peaople with quantified salary: ", quantified_salary_number)
have_an_email = len(df[df["email_address"] != "NaN"])
print("Number of people who have an email: ", have_an_email)

print("Percent of NaN for total payments: ", len(df[df["total_payments"] == "NaN"])/(number_of_people/100))
poi_nan_payments_count = 0

for index, row in df.iterrows():
    if row['poi'] == True:
        if pd.isna(row['total_payments']):
            poi_nan_payments_count += 1
percentage = (poi_nan_payments_count / (number_of_people/100))
print(f"Percent of NaN in POI for total payments: {percentage:.2f}%")