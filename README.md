# Udacity Machine Learning Course Projects

This repository contains all the mini-projects and the final project completed as part of the [Udacity Intro to Machine Learning course](https://www.udacity.com/course/intro-to-machine-learning--ud120).

## Overview

The course covers the fundamentals of machine learning, including:
- Decision trees
- Naive Bayes
- Support Vector Machines (SVM)
- K-Means clustering
- Feature selection and scaling
- Dimensionality reduction (PCA)
- Validation techniques and cross-validation
- Text classification (TF-IDF)
- Building and tuning pipelines

## Final Project: Enron Fraud Detection

The final project builds a **POI (Person of Interest) identifier** using financial and email data from the Enron corpus. Key steps include:

- Feature engineering (including custom features)
- Feature selection via univariate analysis
- Outlier removal
- Model comparison and evaluation (Naive Bayes, Decision Tree, SVM, etc.)
- Parameter tuning using GridSearch

Final model:

DecisionTreeClassifier(
    class_weight='balanced',
    max_depth=6,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

Achieved:

* **Precision > 0.3**
* **Recall > 0.3**

## Notes

* All projects were completed using publicly available Udacity datasets.
* Some datasets might need to be re-downloaded or unzipped manually depending on your system.
* This repository is for educational purposes.
