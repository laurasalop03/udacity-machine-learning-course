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

enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))

print("number data points: ", len(enron_data))
print("features for each person: ", len(enron_data["SKILLING JEFFREY K"]))

for key in enron_data["PRENTICE JAMES"]:
    print ("key: %s , value: %s" % (key, enron_data["PRENTICE JAMES"][key]))

print("Email messages from Wesley Colwell: ", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])

print("Value of stock options by Jeffrey K Skilling: ", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

salary = 0
email = 0
total_payments = 0
total_people = 0
n_poi = 0

for i in enron_data:
    total_people += 1
    if enron_data[i]["salary"] != "NaN":
        salary += 1
    if enron_data[i]["email_address"] != "NaN":
        email += 1
    if enron_data[i]["total_payments"] == "NaN":
        total_payments += 1
    if enron_data[i]["poi"] == 1:
        n_poi += 1

print (total_people, " total people")
print(total_payments, " people with NaN total payments")
print(n_poi, " people of interest")
