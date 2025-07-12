#!/usr/bin/python

import sys
import pickle
import os
sys.path.append(os.path.abspath(("../tools/")))

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
    'poi',
    'salary',
    'bonus',
    'total_payments',
    'exercised_stock_options',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi'
]

### Load the dictionary containing the dataset
with open("./final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)



### Task 2: Remove outliers

## check for outliers in the dataset

# for person in data_dict:
#     print(person)
# there's a "person" called THE TRAVEL AGENCY IN THE PARK, not a real person
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)


## check for outliers graphically

# import matplotlib.pyplot as plt
# for person in data_dict:
#     salary = data_dict[person]['salary']
#     bonus = data_dict[person]['bonus']
#     if salary != "NaN" and bonus != "NaN":
#         plt.scatter(salary, bonus)

# plt.xlabel("Salary")
# plt.ylabel("Bonus")
# plt.show()
# another outlier is the "person" with a salary of 1111258 and a bonus of 7000000, which is much higher than the rest of the dataset
data_dict.pop("TOTAL", 0) 


## check for person with many NaN values

# for person, features in data_dict.items():
#     nan_count = sum([1 for val in features.values() if val == "NaN"])
#     if nan_count > 18: # 21 features in total, so more than 18 NaNs is suspicious
#         print(person, "->", nan_count, "NaNs")

data_dict.pop("LOCKHART EUGENE E", 0)  # 20 NaNs



### Task 3: Create new feature(s)

def compute_fraction(num, den):
    if num == "NaN" or den == "NaN" or den == 0:
        return 0.
    return float(num) / float(den)

for person in data_dict:
    data_point = data_dict[person]
    
    data_point["fraction_from_poi"] = compute_fraction(
        data_point.get("from_poi_to_this_person", "NaN"),
        data_point.get("to_messages", "NaN"))

    data_point["fraction_to_poi"] = compute_fraction(
        data_point.get("from_this_person_to_poi", "NaN"),
        data_point.get("from_messages", "NaN"))

# update features_list to include the new features
features_list = [
    'poi',
    'salary',
    'bonus',
    'total_payments',
    'exercised_stock_options',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'fraction_from_poi',
    'fraction_to_poi'
]


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# keep k best features

from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

k_best = SelectKBest(score_func=f_classif, k='all') 
k_best.fit(features, labels)

scores = k_best.scores_
pvalues = k_best.pvalues_
features_scores = list(zip(features_list[1:], scores, pvalues))  # without 'poi'

# order from highest to lowest score
features_scores = sorted(features_scores, key=lambda x: x[1], reverse=True)

# print("\n### Feature ranking:")
# for i, (feature, score, pval) in enumerate(features_scores, 1):
#     print(f"{i}. {feature:30} score: {score:.2f}   p-value: {pval:.5f}")

### Feature ranking:
# 1. exercised_stock_options        score: 24.25   p-value: 0.00000
# 2. bonus                          score: 20.26   p-value: 0.00001
# 3. salary                         score: 17.72   p-value: 0.00005
# 4. fraction_to_poi                score: 15.95   p-value: 0.00011
# 5. total_payments                 score: 8.57   p-value: 0.00399
# 6. shared_receipt_with_poi        score: 8.28   p-value: 0.00465
# 7. from_poi_to_this_person        score: 5.04   p-value: 0.02633
# 8. fraction_from_poi              score: 2.96   p-value: 0.08736
# 9. from_this_person_to_poi        score: 2.30   p-value: 0.13205


# keep the features with pvalue < 0.05
features_list = [
    'poi',
    'exercised_stock_options',
    'bonus',
    'salary',
    'fraction_to_poi',
    'total_payments',
    'shared_receipt_with_poi',
    'from_poi_to_this_person'
]



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

param_grids = {
    "Naive Bayes": [{}],  
    
    "Decision Tree": [
        {"max_depth": 2},
        {"max_depth": 4},
        {"max_depth": 6},
        {"max_depth": None, "class_weight": "balanced"},
    ],
    
    "K-Nearest Neighbors": [
        {"n_neighbors": 1},
        {"n_neighbors": 3},
        {"n_neighbors": 5},
        {"n_neighbors": 7},
    ],
    
    "SVM": [
        {"C": 1, "gamma": "scale", "class_weight": "balanced"},
        {"C": 10, "gamma": "scale", "class_weight": "balanced"},
        {"C": 100, "gamma": "scale", "class_weight": "balanced"},
        {"C": 10, "gamma": 0.001, "class_weight": "balanced"},
    ]
}

# for clf_name, params_list in param_grids.items():
#     print(f"== {clf_name} ==")
#     for params in params_list:
#         if clf_name == "Naive Bayes":
#             clf = GaussianNB()
#         elif clf_name == "Decision Tree":
#             clf = DecisionTreeClassifier(random_state=42, **params)
#         elif clf_name == "K-Nearest Neighbors":
#             clf = KNeighborsClassifier(**params)
#         elif clf_name == "SVM":
#             clf = SVC(kernel='rbf', random_state=42, **params)
        
#         clf.fit(features_train, labels_train)
#         predictions = clf.predict(features_test)
        
#         acc = accuracy_score(labels_test, predictions)
#         prec = precision_score(labels_test, predictions, zero_division=0)
#         rec = recall_score(labels_test, predictions, zero_division=0)
        
#         print(f"Params: {params} -> Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")
#     print()

# == Naive Bayes ==
# Params: {} -> Accuracy: 0.837, Precision: 0.250, Recall: 0.667

# == Decision Tree ==
# Params: {'max_depth': 2} -> Accuracy: 0.860, Precision: 0.200, Recall: 0.333
# Params: {'max_depth': 4} -> Accuracy: 0.884, Precision: 0.250, Recall: 0.333
# Params: {'max_depth': 6} -> Accuracy: 0.907, Precision: 0.400, Recall: 0.667     <-- the best one
# Params: {'max_depth': None, 'class_weight': 'balanced'} -> Accuracy: 0.884, Precision: 0.000, Recall: 0.000

# == K-Nearest Neighbors ==
# Params: {'n_neighbors': 1} -> Accuracy: 0.837, Precision: 0.000, Recall: 0.000
# Params: {'n_neighbors': 3} -> Accuracy: 0.907, Precision: 0.000, Recall: 0.000
# Params: {'n_neighbors': 5} -> Accuracy: 0.930, Precision: 0.000, Recall: 0.000
# Params: {'n_neighbors': 7} -> Accuracy: 0.930, Precision: 0.000, Recall: 0.000

# == SVM ==
# Params: {'C': 1, 'gamma': 'scale', 'class_weight': 'balanced'} -> Accuracy: 0.721, Precision: 0.000, Recall: 0.000
# Params: {'C': 10, 'gamma': 'scale', 'class_weight': 'balanced'} -> Accuracy: 0.605, Precision: 0.000, Recall: 0.000
# Params: {'C': 100, 'gamma': 'scale', 'class_weight': 'balanced'} -> Accuracy: 0.674, Precision: 0.000, Recall: 0.000
# Params: {'C': 10, 'gamma': 0.001, 'class_weight': 'balanced'} -> Accuracy: 0.930, Precision: 0.000, Recall: 0.000


# we choose the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [4, 6, 8, None],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': [None, 'balanced']
}

grid_search = GridSearchCV(clf, param_grid, scoring='f1', cv=10, n_jobs=-1)
grid_search.fit(features_train, labels_train)

results = grid_search.cv_results_
params_list = results['params']
mean_test_scores = results['mean_test_score']

# order the results by mean test score to get the best combinations
sorted_indices = mean_test_scores.argsort()[::-1]

# print("Top 10 combinations of param:")
# for i in sorted_indices[:10]:
#     print(f"Param: {params_list[i]}, F1 Score: {mean_test_scores[i]:.4f}")


# after trying different parameters, the following combination gives the best results in tester.py
# # DecisionTreeClassifier(class_weight='balanced', max_depth=6, random_state=42)
# #         Accuracy: 0.86147       Precision: 0.48028      Recall: 0.47500 F1: 0.47763     F2: 0.47605
# #         Total predictions: 15000        True positives:  950    False positives: 1028   False negatives: 1050   True negatives: 11972

clf = DecisionTreeClassifier(max_depth=6, min_samples_split=2, min_samples_leaf=1, class_weight='balanced', random_state=42)
clf.fit(features_train, labels_train)




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
