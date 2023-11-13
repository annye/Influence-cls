

"""
Desc: The purpose of this script is to return the paramater search space for each ML algorithm.
      The space is defined in such a way as to be used with the hyperopt parameter tuning package.
	  
"""

import numpy as np
from hyperopt import hp

def return_parameter_space(algo):
    """Return parameter space for each algo."""
    
    space = {}

    if algo == 'LR':
        # space['solver'] = hp.choice('solver', ['liblinear'])
        # space['C'] = hp.loguniform('C', np.log(0.00001), np.log(100))
	space['solver'] = hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
        space['penalty'] = hp.choice('penalty', [ 'l2'])
        space['C'] = hp.loguniform('C', np.log(0.00001), np.log(100))
        space['fit_intercept'] = hp.choice('fit_intercept', [True, False])
        space['intercept_scaling'] = hp.uniform('intercept_scaling', 0.1, 2.0)
        space['l1_ratio'] = hp.uniform('l1_ratio', 0.0, 1.0)
        space['class_weight'] = hp.choice('class_weight', [None, 'balanced'])
      

    elif algo == 'SVM':
        space['kernel'] = hp.choice('kernel', ['linear', 'rbf'])
        space['gamma'] = hp.choice('gamma', ['scale', 'auto'])
        space['C'] = hp.loguniform('C', np.log(0.00001), np.log(100))
    elif algo == 'NB':
        space = {'var_smoothing': hp.loguniform('learning_rate', -20, 0)}
    elif algo == 'RF':
        space['n_estimators'] = hp.choice('n_estimators', [50, 100, 200, 300, 400, 500])
        space['max_depth'] = hp.choice('max_depth', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        space['min_samples_split'] = hp.choice('min_samples_split', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        space['min_samples_leaf'] = hp.choice('min_samples_leaf', [0.1, 0.2, 0.3, 0.4, 0.5])
        space['max_features'] = hp.choice('max_features', ['sqrt', 'log2', None])
        space['bootstrap'] = hp.choice('bootstrap', [True, False])
    elif algo == 'KNN':
        space = {
            'n_neighbors': hp.choice('n_neighbors', range(1, 21)),
            'p': hp.choice('p', [1, 2]),
            'weights': hp.choice('weights', ['uniform', 'distance']),
            'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        }
    elif algo == 'XGB':
        space = {
            'n_estimators': hp.choice('n_estimators', [50, 100, 200, 300, 400, 500]),
            'booster': hp.choice('booster', ['gbtree']),
            'tree_method': hp.choice('tree_method', ['auto', 'hist', 'approx']),
            'learning_rate': hp.loguniform('learning_rate', -5, 0),
            'max_depth': hp.choice('max_depth', np.arange(3, 10, dtype=int)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'gamma': hp.uniform('gamma', 0, 1),
            'lambda': hp.uniform('lambda', 0, 5),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1)
        }
    elif algo == 'LGBM':
        space = {
            'n_estimators': hp.choice('n_estimators', [50, 100, 200, 300, 400, 500]),
            'learning_rate': hp.loguniform('learning_rate', -5, 0),
            'max_depth': hp.choice('max_depth', np.arange(3, 10, dtype=int)),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'reg_alpha': hp.uniform('reg_alpha', 0, 1),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1)
        }

    return space
