#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import os
import joblib
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb") )

features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(features_train, labels_train)
print("accuracy:", tree.score(features_test, labels_test))

n_poi_test_set = sum(labels_test)
print("Number of POIs in the test set:", n_poi_test_set)

n_people_test_set = len(labels_test)
print("Number of people in the test set:", n_people_test_set)

n_predicted_pois = sum(tree.predict(features_test))
print("Number of predicted POIs:", n_predicted_pois)

true_pos = sum((labels_test[i] == 1 and tree.predict(features_test[i].reshape(1, -1)) == 1) for i in range(len(labels_test)))
print("True Positives:", true_pos)

# for i in len(labels_test):
#     if labels_test[i] == 1 and tree.predict(features_test[i].reshape(1, -1)) == 1:
#         print("True Positive at index", i)
#     elif labels_test[i] == 0 and tree.predict(features_test[i].reshape(1, -1)) == 1:
#         print("False Positive at index", i)
#     elif labels_test[i] == 1 and tree.predict(features_test[i].reshape(1, -1)) == 0:
#         print("False Negative at index", i)
#     elif labels_test[i] == 0 and tree.predict(features_test[i].reshape(1, -1)) == 0:
#         print("True Negative at index", i)

from sklearn.metrics import precision_score, recall_score
precision = precision_score(labels_test, tree.predict(features_test))
recall = recall_score(labels_test, tree.predict(features_test))
print("Precision:", precision)
print("Recall:", recall)

# precision and recall = 0 because true positives n_poi = 0
