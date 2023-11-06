# Standard imports
import numpy as np
import pandas as pd
from scipy import stats
from pdb import set_trace
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import binarize
from collections import Counter
# Data loading and preprocessing
from dataloading import *
from get_parameter_space import *
from assess_performance import *
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from lightgbm import *
# Data splitting
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline # Pipeline
from imblearn.pipeline import Pipeline
# Evaluation metrics
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.metrics import auc
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import f1_score
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
# SHAP (SHapley Additive exPlanations)
import shap
from shaphypetune import BoostSearch, BoostRFE, BoostRFA, BoostBoruta
from hyperopt import hp
from hyperopt import Trials

import hyperopt
from hyperopt import fmin, rand, tpe, hp, Trials, space_eval
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
import time
hyperopt_rstate = np.random.RandomState(42)
import warnings
# Filter and ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="validation_0-logloss")


def build_models(X, y, algo, key):
    """Construct ML models."""
    
    es_rounds = 0

    def objective_xgb(params):
        model = XGBClassifier(random_state=42, **params)
        # Use early stopping with XGBoost
        pipeline = Pipeline([('ROS', RandomOverSampler(random_state=42)),
                            ('model', model)])
        early_stopping_params = {'model__eval_metric': 'auc',
                                 'model__early_stopping_rounds': 10,  # You can adjust this number
                                 'model__verbose': False,
                                 'model__eval_set': [[X_val, y_val]]}
        cv_inner = StratifiedKFold(n_splits=3, shuffle=False)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv_inner,
                                 scoring='roc_auc', error_score='raise',
                                 fit_params=early_stopping_params) 
        mean_auc = np.mean(scores)
        return -mean_auc


    def objective_lr(params):
        scaler = StandardScaler()
        model = LogisticRegression(random_state=42, max_iter=100, n_jobs=-1, **params) #, max_iter=5000
        pipeline = Pipeline([('scaler', scaler), 
                            ('ROS', RandomOverSampler(random_state=42)),
                            ('model', model)])
        cv_inner = StratifiedKFold(n_splits=3, shuffle=False)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv_inner,
                                 scoring='roc_auc', error_score='raise') 
        mean_auc = np.mean(scores)
        return -mean_auc
    
    def objective_svm(params):
        scaler = StandardScaler()
        model = SVC(random_state=42, probability=True, **params)
        pipeline = Pipeline([('scaler', scaler), 
                            ('ROS', RandomOverSampler(random_state=42)),
                            ('model', model)]
                            )

        cv_inner = StratifiedKFold(n_splits=3, shuffle=False)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv_inner, scoring='roc_auc', error_score='raise')
        mean_auc = np.mean(scores)

        return -mean_auc
    
    def objective_nb(params):
        scaler = StandardScaler()
        model = GaussianNB(**params)
        pipeline = Pipeline([('scaler', scaler), 
                            ('ROS', RandomOverSampler(random_state=42)),
                            ('model', model)]
                            )

        cv_inner = StratifiedKFold(n_splits=3, shuffle=False)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv_inner, scoring='roc_auc', error_score='raise')
        mean_auc = np.mean(scores)

        return -mean_auc
    
    def objective_rf(params):
        params['n_estimators'] = int(params['n_estimators'])
        scaler = StandardScaler()
        model = RandomForestClassifier(random_state=42, **params)
        pipeline = Pipeline([('scaler', scaler), 
                            ('ROS', RandomOverSampler(random_state=42)),
                            ('model', model)]
                            )

        cv_inner = StratifiedKFold(n_splits=3, shuffle=False)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv_inner, scoring='roc_auc', error_score='raise')
        mean_auc = np.mean(scores)

        return -mean_auc




    start = time.time()
    obj_fns = {'LR': objective_lr, 'SVM': objective_svm, 'RF': objective_rf,
               'XGB': objective_xgb, 'NB': objective_nb}

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    # Get train/test splits
    outer_predictions = {'Fold predictions': [], 'Fold probabilities': [], 'Fold test': []}
    cv_outer = StratifiedKFold(n_splits=10, shuffle=False)
    
    for train_ix, test_ix in cv_outer.split(X,y):
        # split data
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        if algo == 'XGB':
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
        space = return_parameter_space(algo)
        trials = Trials()
        
        best = fmin(fn=obj_fns[algo], space=space, algo=tpe.suggest, max_evals=100, trials=trials, rstate=hyperopt_rstate) 
        
        # Retrieve the best parameters
        best_params = space_eval(space, best)
        if algo in ['LR']:
            best_model = LogisticRegression(random_state=42, n_jobs=-1, **best_params) # tree_method='hist', 
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            ros = RandomOverSampler()
            X_train, y_train = ros.fit_resample(X_train, y_train)
            best_model.fit(X_train, y_train)
            
        # --------------------------------
        elif algo == 'SVM':
            best_model = SVC(random_state=42, probability=True, **best_params)  # Initialize the model
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)  # Scale your training data
            ros = RandomOverSampler(random_state=42)
            X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)  # Resample the training data if needed
            best_model.fit(X_train_resampled, y_train_resampled)  # Fit the model on the training data
            y_pred = best_model.predict(X_test)  # Make predictions on the test data
            y_probas = best_model.predict_proba(X_test)[:, 1]  # Get predicted probabilities


        # --------------------------------
        elif algo == 'RF':
            best_model = RandomForestClassifier(random_state=42, **best_params)  # Initialize the model
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)  # Scale your training data
            ros = RandomOverSampler(random_state=42)
            X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)  # Resample the training data if needed
            best_model.fit(X_train_resampled, y_train_resampled)  # Fit the model on the training data
            y_pred = best_model.predict(X_test)  # Make predictions on the test data
            y_probas = best_model.predict_proba(X_test)[:, 1]  # Get predicted probabilities

       
        # --------------------------------
        elif algo == 'NB':
            best_model = GaussianNB(**best_params)  # Initialize the model
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)  # Scale your training data
            ros = RandomOverSampler(random_state=42)
            X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)  # Resample the training data if needed
            best_model.fit(X_train_resampled, y_train_resampled)  # Fit the model on the training data
            y_pred = best_model.predict(X_test)  # Make predictions on the test data
            y_probas = best_model.predict_proba(X_test)[:, 1]  # Get predicted probabilities

        # --------------------------------
        elif algo == 'XGB':
            best_model = XGBClassifier(random_state=42, **best_params)
            ros = RandomOverSampler()
            X_train, y_train = ros.fit_resample(X_train, y_train)
            eval_set = [(X_val, y_val)]  # Use a validation set for early stopping
            es_rounds = 10
            best_model.fit(X_train, y_train,
                           early_stopping_rounds=es_rounds,
                           eval_set=eval_set,
                           verbose=True)
        # --------------------------------
        elif algo == 'LGBM':
            best_model = LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    
        # evaluate model on the hold out dataset
        y_pred = best_model.predict(X_test)
        # Get predicted probabilities
        y_probas = best_model.predict_proba(X_test)[::, 1] 
        outer_predictions['Fold predictions'].append((y_pred)); 
        outer_predictions['Fold probabilities'].append((y_probas))
        outer_predictions['Fold test'].append((y_test))
    
    # Summarize the estimated performance of the model over nested CV outer test sets
    # results = get_and_record_scores(outer_predictions)
    # save_results_dictionary(results, 'results_' + str(algo) + '_' + str(key) + '_' + str(es_rounds) + '_hyperopt.pkl')        
    # print("Duration for {}: {}".format(str(algo), time.time() - start))
    # print(key)
    # print(results)
    results = get_and_record_scores(outer_predictions)

    # Assuming that `results` is a dictionary
    result_df = pd.DataFrame.from_dict(results, orient='index')

    # Define the CSV file name
    csv_file_name = 'results_' + str(algo) + '_' + str(key) + '_' + str(es_rounds) + '_hyperopt.csv'

    # Save the DataFrame to a CSV file
    result_df.to_csv(csv_file_name)

    # Save the original dictionary as a pickle file
    save_results_dictionary(results, 'results_' + str(algo) + '_' + str(key) + '_' + str(es_rounds) + '_hyperopt.pkl')

    print("Duration for {}: {}".format(str(algo), time.time() - start))
    print(key)
    print(result_df)
    
