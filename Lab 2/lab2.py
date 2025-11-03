#!/usr/bin/env python3
"""lab2 template"""
import numpy as np
from numpy import ravel
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from common import feature_selection as feat_sel
from common import test_env


# Pylint error W0621: Redefining name 'X' from outer scope (line 61) (redefined-outer-name)
# pylint: disable=redefined-outer-name


def print_metrics(y_true, y_pred, label):
    # Feel free to extend it with additional metrics from sklearn.metrics
    print('%s R squared: %.2f' % (label, r2_score(y_true, y_pred)))


def linear_regression(X, y, print_text='Linear regression all in'):
    # Split train test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # Linear regression all in
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    print_metrics(y_test, reg.predict(X_test), print_text)
    return reg


def linear_regression_selection(X, y):
    X_sel = feat_sel.backward_elimination(X, y)
    return linear_regression(
        X_sel, y, print_text='Linear regression with feature selection')


# TODO: ADD POLYNOMIAL REGRESSION
def polynomial_regression(X, y):
    poly_reg = PolynomialFeatures(degree=2)
    X_poly = poly_reg.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    return linear_regression(
        X_poly, y, print_text='Polynomial regression with feature selection')


# TODO: ADD SVR
def support_vector_regression(X, y):
    sc_X = StandardScaler()
    sc_Y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_Y.fit_transform(np.expand_dims(y, axis=1))
    #y = sc_Y.fit_transform(y)
    # TODO: STUDENT SHALL ADD TEST-TRAIN SPLIT AND REGRESSOR CREATION AND TRAINING HERE
    # Test-Train Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, ravel(y), test_size=0.25, random_state=0)
    # Regressor creation
    reg = SVR(kernel='rbf', gamma='auto')
    # Training
    reg.fit(X_train, y_train)
    #reg.fit(X_train, np.squeeze(y_train, axis=1))

    print_metrics(np.squeeze(y_test), np.squeeze(reg.predict(X_test)), 'SVR')


# TODO: ADD DECISION TREE REGRESSION REGRESSION
def decision_tree_regression(X, y):
    # Test-Train Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    # Regressor creation
    #reg = DecisionTreeRegressor(random_state=1)
    reg = DecisionTreeRegressor(
        random_state=0,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1)
    # Training
    reg.fit(X_train, y_train)

    print_metrics(
        np.squeeze(y_test),
        np.squeeze(
            reg.predict(X_test)),
        'Decision Tree Regression')


# TODO: ADD RANDOM FOREST REGRESSION
def random_forest_regression(X, y):
    # Test-Train Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    # Regressor creation
    reg = RandomForestRegressor(n_estimators=10, random_state=0)
    # Training
    reg.fit(X_train, y_train)
    # print result
    print_metrics(
        np.squeeze(y_test),
        np.squeeze(
            reg.predict(X_test)),
        'Random Forest Regression')


if __name__ == '__main__':
    test_env.versions(['numpy', 'statsmodels', 'sklearn'])

    # https://scikit-learn.org/stable/datasets/index.html#boston-house-prices-dataset
    X, y = load_boston(return_X_y=True)

    linear_regression(X, y)
    linear_regression_selection(X, y)
    # TODO: CALL POLYNOMIAL REGRESSION
    polynomial_regression(X, y)
    # TODO: CALL SVR
    support_vector_regression(X, y)
    # TODO: CALL DECISION TREE REGRESSION
    decision_tree_regression(X, y)
    # TODO: CALL RANDOM FOREST REGRESSION FUNCTIONS
    random_forest_regression(X, y)
    print('Done')
