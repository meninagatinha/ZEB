import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from collections import defaultdict


class ModelRunner:
    """
    A container class for running Random Forest Regression, Support Vector Regression, and XGBoost

    Grid Search params are set as class attributes, too lazy to put them in a config file
    """

    VERBOSE, N_JOBS, CV = 2, 4, 3
    RF_N_ITER = 100
    RF_MAX_DEPTH = [int(x) for x in np.linspace(10, 110, num = 11)] + [None]
    N_ESTIMATORS = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    MAX_FEATURES = [None]
    MIN_SAMPLES_SPLIT = [2, 5, 10]
    MIN_SAMPLES_LEAF = [1, 2, 4]
    RF_RANDOMIZED_SEARCH_GRID_PARAMS = {"n_estimators": N_ESTIMATORS, \
                                        "max_features": MAX_FEATURES, \
                                        "max_depth": RF_MAX_DEPTH, \
                                        "min_samples_split": MIN_SAMPLES_SPLIT, \
                                        "min_samples_leaf": MIN_SAMPLES_LEAF, \
                                        "bootstrap": [True, False]}
    SVR_GRID_SEARCH_PARAMS = {"kernel":["rbf", "sigmoid"], \
                              "C": np.logspace(np.log10(0.001), np.log10(200), num=10), \
                              "gamma": np.logspace(np.log10(0.00001), np.log10(2), num=20)}
    XGBOOST_GRID_SEARCH_PARAMS = {"objective": ["reg:squarederror"], \
                                  "learning_rate": [0.1], \
                                  "n_estimators": N_ESTIMATORS, \
                                  "max_depth": RF_MAX_DEPTH, \
                                  "min_child_weight": [4], \
                                  "colsample_bytree": [0.7]}

    def __init__(self, Y, X, metric=metrics.r2_score, test_size=0.2, random_state=42):
        """
        Parameters
        ----------
        Y : DataFrame, Matrix, or Series containing dependent variable(s)
        X : DataFrame containing covariates
        metric: A regression evaluation metric from sklearn.metrics. Default is R-squared
        test_size: float between (0, 1) specifying proportion of test data. Default is 0.2
        random_state: int specifying random state
        """
        self.X, self.Y, self.metric, self.random_state = X, Y, metric, random_state
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X,
                                                                                Y,
                                                                                test_size=test_size,
                                                                                random_state=self.random_state)
        self.models = {}
        self.model_scores = {}

    def run_random_forest_regression(self):
        """
        Trains random forest on the training set using randomized search cross validation.
        Selects best estimator and adds to models dictionary

        Args set to default in class attributes
        """
        rf = RandomForestRegressor()
        rf_random = RandomizedSearchCV(estimator=rf, 
                                       param_distributions=self.RF_RANDOMIZED_SEARCH_GRID_PARAMS,
                                       n_iter=self.RF_N_ITER, 
                                       cv=self.CV, 
                                       verbose=self.VERBOSE, 
                                       random_state=self.random_state, 
                                       n_jobs=self.N_JOBS)
        rf_random.fit(self.X_train, self.Y_train)
        self.models["RF"] = rf_random.best_estimator_

    def run_svr(self):
        """
        Trains SVR on the training set using grid search cross validation.
        Selects best estimator and adds to models dictionary

        Args set to default in class attributes
        """
        svr = SVR()
        grid_searcher = GridSearchCV(svr, self.SVR_GRID_SEARCH_PARAMS, n_jobs=self.N_JOBS, verbose=self.VERBOSE, cv=self.CV)
        grid_searcher.fit(self.X_train, self.Y_train)
        self.models["SVR"] = grid_searcher.best_estimator_

    def run_xgboost(self):
        """
        Trains XGBoost on the training set using grid search cross validation.
        Selects best estimator and adds to models dictionary

        Args set to default in class attributes
        """
        xgb = XGBRegressor()
        xgb_grid_search = GridSearchCV(xgb, self.XGBOOST_GRID_SEARCH_PARAMS, cv=self.CV, n_jobs=self.N_JOBS, verbose=True)
        xgb_grid_search.fit(self.X_train, self.Y_train)
        self.models["XGBoost"] = xgb_grid.best_estimator_

    def score_models(self):
        """
        Scores the models with self.metric  on both the training and test datasets.

        Scores are available in public attribute model_scores
        """
        for name, model in self.models.items():
            train_score = self.metric(self.Y_train, model.predict(self.X_train))
            test_score = self.metric(self.Y_test, model.predict(self.X_test))
            self.model_scores[name] = (train_score, test_score)
    