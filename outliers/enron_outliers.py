#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data_dict.pop("TOTAL", 0)  # Remove the "TOTAL" entry, as it is an outlier
data = featureFormat(data_dict, features)


### your code below
for fearture, target in data:
    matplotlib.pyplot.scatter(fearture, target)

matplotlib.pyplot.xlabel(features[0])
matplotlib.pyplot.ylabel(features[1])
matplotlib.pyplot.title("Enron Salary vs Bonus")
matplotlib.pyplot.show()

