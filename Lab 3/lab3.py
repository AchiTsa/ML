#!/usr/bin/env python3
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# TODO
#   STUDENT SHALL ADD NEEDED IMPORTS

from common import describe_data, test_env
from common.classification_metrics import print_metrics


def read_data(file):
    """Return pandas dataFrame read from Excel file"""
    try:
        return pd.read_excel(file)
    except FileNotFoundError:
        sys.exit('ERROR: ' + file + ' not found')


def preprocess_data(df, verbose=False):
    y_column = 'In university after 4 semesters'

    # Features can be excluded by adding column name to list
    drop_columns = []

    categorical_columns = [
        'Faculty',
        'Paid tuition',
        'Study load',
        'Previous school level',
        'Previous school study language',
        'Recognition',
        'Study language',
        'Foreign student'
    ]

    # Handle dependent variable
    if verbose:
        print('Missing y values: ', df[y_column].isna().sum())

    y = df[y_column].values
    # Encode y. Naive solution
    y = np.where(y == 'No', 0, y)
    y = np.where(y == 'Yes', 1, y)
    y = y.astype(float)

    # Drop also dependent variable column to leave only features
    drop_columns.append(y_column)
    df = df.drop(labels=drop_columns, axis=1)

    # Remove drop columns for categorical columns just in case
    categorical_columns = [
        i for i in categorical_columns if i not in drop_columns]

    # TODO
    #   STUDENT SHALL HANDLE MISSING VALUES

    # replace missing categorical values with common label
    for column in categorical_columns:
        df[column] = df[column].fillna(value='Missing')

    # TODO
    #   STUDENT SHALL ENCODE CATEGORICAL FEATURES
    for column in categorical_columns:
        df = pd.get_dummies(df, prefix=[column], columns=[column])

    # Handle missing data. At this point only exam points should be missing
    # It seems to be easier to fill whole data frame as only particular columns
    if verbose:
        describe_data.print_nan_counts(df)

    # Replace rest of missing values (missing exam points) with 0.
    df['Estonian language exam points'] = df['Estonian language exam points'].fillna(
        value=0)
    df['Estonian as second language exam points'] = df['Estonian as second language exam points'].fillna(
        value=0)
    df['Mother tongue exam points'] = df['Mother tongue exam points'].fillna(
        value=0)
    df['Narrow mathematics exam points'] = df['Narrow mathematics exam points'].fillna(
        value=0)
    df['Wide mathematics exam points'] = df['Wide mathematics exam points'].fillna(
        value=0)
    df['Mathematics exam points'] = df['Mathematics exam points'].fillna(
        value=0)

    if verbose:
        describe_data.print_nan_counts(df)

    # Return features data frame and dependent variable
    return df, y


# TODO
#   LOGISTIC REGRESSION CLASSIFIER
def logistic_regression_classifier(X, y):
    title = 'LOGISTIC REGRESSION CLASSIFIER'
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = LogisticRegression(random_state=0)  # , solver='saga')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print_metrics(y_test, y_pred, label=title)
    print("Precision Score is ", precision_score(y_test, y_pred) * 100, "%.")
    print()


# TODO
#   KNN CLASSIFIER
def knn_classifier(X, y):
    title = 'KNN CLASSIFIER'

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = KNeighborsClassifier(n_neighbors=6, p=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print_metrics(y_test, y_pred, label=title)
    print("Precision Score is ", precision_score(y_test, y_pred) * 100, "%.")
    print()


# TODO
#   SVM CLASSIFIER
def SVM_classifier(X, y):
    title = 'SVM CLASSIFIER'

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = SVC(
        kernel='sigmoid',
        gamma=0.2,
        tol=1e-2,
        probability=True,
        random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print_metrics(y_test, y_pred, label=title)
    print("Precision Score is ", precision_score(y_test, y_pred) * 100, "%.")
    print()


# TODO
#   NAIVE BAYES CLASSIFIER
def naive_bayes_classifier(X, y):
    title = 'NAIVE BAYES CLASSIFIER'

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    #sc = StandardScaler()
    #X_train = sc.fit_transform(X_train)
    #X_test = sc.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print_metrics(y_test, y_pred, label=title)
    print("Precision Score is ", precision_score(y_test, y_pred) * 100, "%.")
    print()


# TODO
#   DECISION TREE CLASSIFIER
def decision_tree_classifier(X, y):
    title = 'DECISION TREE CLASSIFIER'

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print_metrics(y_test, y_pred, label=title)
    print("Precision Score is ", precision_score(y_test, y_pred) * 100, "%.")
    print()


# TODO
#   RANDOM FOREST CLASSIFIER
def random_forest_classifier(X, y):
    title = 'RANDOM FOREST CLASSIFIER'

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print_metrics(y_test, y_pred, label=title)
    print("Precision Score is ", precision_score(y_test, y_pred) * 100, "%.")
    print()


if __name__ == '__main__':
    modules = ['numpy', 'pandas', 'sklearn']
    test_env.versions(modules)

    students = read_data('data/students.xlsx')

    # TODO
    #   DONE
    #   STUDENT SHALL CALL PRINT_OVERVIEW AND PRINT_CATEGORICAL FUNCTIONS WITH
    #   FILE NAME AS ARGUMENT
    describe_data.print_overview(
        students, file='results/students_overview.txt')
    describe_data.print_categorical(
        students, file='results/students_categorical_data.txt')
    dropped_students = students[(
        students['In university after 4 semesters'] == 'No')]
    describe_data.print_overview(dropped_students,
                                 file='results/dropped_students_overview.txt')
    describe_data.print_categorical(
        dropped_students,
        file='results/dropped_students_categorical_data.txt')

    students_X, students_y = preprocess_data(students)

    wcss = []
    for number_of_clusters in range(1, 25):
        sc = StandardScaler()
        X_train = sc.fit_transform(students_X)
        X_test = sc.transform(students_X)

        kmeans = KMeans(n_clusters=number_of_clusters, random_state=0)
        kmeans.fit(students_X, students_y)
        wcss.append(kmeans.inertia_)
    ks = list(range(1, 25))
    plot_graph = plt.plot(ks, wcss)
    plt.show()



    # TODO
    #   STUDENT SHALL CALL CREATED CLASSIFIERS FUNCTIONS
    logistic_regression_classifier(students_X, students_y)
    knn_classifier(students_X, students_y)
    SVM_classifier(students_X, students_y)
    naive_bayes_classifier(students_X, students_y)
    decision_tree_classifier(students_X, students_y)
    random_forest_classifier(students_X, students_y)

    print('Done')
