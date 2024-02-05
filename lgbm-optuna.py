from load_data import *
import numpy as np
import pandas as pd
from pdb import set_trace
import shap
# Filter out warnings
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold


# Split the data into train and validation sets for each target separately

(X, X_train1, X_val1, y_train1, y_val1,
 X_train2, X_val2, y_train2, y_val2,
 X_train3, X_val3, y_train3, y_val3) = get_data()


# Split the data into train, validation, and test sets for hyperparameter optimization
X_train1, X_valid1, y_train1, y_valid1 = train_test_split(X_train1, y_train1, test_size=0.2, random_state=23)
X_train2, X_valid2, y_train2, y_valid2 = train_test_split(X_train2, y_train2, test_size=0.2, random_state=23)
X_train3, X_valid3, y_train3, y_valid3 = train_test_split(X_train3, y_train3, test_size=0.2, random_state=23)


# Define the objective function for Optuna optimization
def lgbm_objective(trial, X_train, y_train, X_valid, y_valid):
    """
    Objective function for Optuna optimization. It trains a LightGBM classifier
    with hyperparameters suggested by Optuna and returns the ROC AUC score on
    the validation set.

    Parameters:
    - trial: Optuna Trial object for hyperparameter tuning.
    - X_train: Training features.
    - y_train: Training labels.
    - X_valid: Validation features.
    - y_valid: Validation labels.

    Returns:
    - roc_auc: ROC AUC score on the validation set.
    """
    params = {
        'objective': 'binary',
        'boosting_type': 'dart',
        'num_leaves': trial.suggest_int('num_leaves', 2, 1000),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 1000),
        'subsample': trial.suggest_float('subsample', 0.1, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.99),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1),
        'n_estimators': 2000,
        'random_state': 23,
        'class_weight': 'balanced',
    }

    clf = lgb.LGBMClassifier(**params, early_stopping_rounds=50, verbose_eval=False)
    clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='auc')
    y_pred_valid = clf.predict_proba(X_valid)[:, 1]
    roc_auc = roc_auc_score(y_valid, y_pred_valid)

    return roc_auc



# Define the number of splits and create StratifiedKFold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=23)

final_classifiers = []
roc_auc_scores = {}

# Store feature importance for each model
feature_importance = {}

# Store metrics for each model
metrics_dict = {}

# Define data splits for each target
data_splits = [
    (X_train1, y_train1, X_valid1, y_valid1),
    (X_train2, y_train2, X_valid2, y_valid2),
    (X_train3, y_train3, X_valid3, y_valid3)
]

for i, (X_train, y_train, X_valid, y_valid) in enumerate(data_splits):
    # Create an Optuna study and optimize the LightGBM model for the current target
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: lgbm_objective(trial, X_train, y_train, X_valid, y_valid), n_trials=1000)
    
    # Get the best parameters for the current target and create the final LightGBM model
    best_params = study.best_params
    final_clf = lgb.LGBMClassifier(**best_params, n_estimators=1000)
    
    # Fit the final model to the entire training dataset for the current target
    final_clf.fit(X_train, y_train)
    
    final_classifiers.append(final_clf)

    # Make predictions for the current target's validation set
    y_pred_val = final_clf.predict_proba(X_valid)[:, 1]
    
    # Calculate ROC AUC score on the validation set for the current target
    roc_auc = round(roc_auc_score(y_valid, y_pred_val), 3)
    
    # Store the ROC AUC score for this target with a label
    roc_auc_scores[f"Model {i+1}"] = roc_auc
    
    print(f"Model {i+1}: ROC AUC = {roc_auc}")
    from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, precision_score

    # Calculate confusion matrix and metrics for the current model
    cm = confusion_matrix(y_valid, (y_pred_val > 0.5).astype(int))
    Recall = cm[1][1] / (cm[1][1] + cm[1][0])
    Specificity = cm[0][0] / (cm[0][0] + cm[0][1])

    # Calculate F1-score for binary classification (two classes)
    F1_Binary = f1_score(y_valid, (y_pred_val > 0.5).astype(int))

    Balanced_Accuracy = balanced_accuracy_score(y_valid, (y_pred_val > 0.5).astype(int))
    Precision = precision_score(y_valid, (y_pred_val > 0.5).astype(int))

    metrics_dict[f"Model {i+1}"] = {
        'Classifier': 'LGBM',
        'Target_Value': f'tech10-d{i+1}',
        'TP': cm[1][1],
        'FP': cm[0][1],
        'TN': cm[0][0],
        'FN': cm[1][0],
        'Recall': Recall,
        'Specificity': Specificity,
        'Precision': Precision,
        'F1_Binary': F1_Binary,  # Use F1_Binary instead of F1_Weighted
        'Balanced_Accuracy': Balanced_Accuracy,
        'roc_auc_val': roc_auc
    }


  

# Example of accessing metrics for Model 1
print("Metrics for Model 1:")
print(metrics_dict["Model 1"])


# Convert the metrics_dict to a DataFrame
metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')

# Reset the index to have 'Model' as a regular column
metrics_df.reset_index(inplace=True)
metrics_df.rename(columns={'index': 'Model'}, inplace=True)

# Display the DataFrame
print(metrics_df)

metrics_df.to_csv('/Users/d18127085/Desktop/statistical_phaseII/src/results/tech10_combined_data2.csv', index=False)
# Define a dictionary to store feature importance for each model
# Define a dictionary to store feature importance for each model
feature_importance = {}

for i, final_clf in enumerate(final_classifiers):
    # Calculate feature importance for the current model and keep only the top 10 features
    feature_imp = pd.DataFrame(sorted(zip(final_clf.feature_importances_, X.columns), reverse=True)[:10],
                               columns=['Value', 'Feature'])
    
    # Store the top 10 features for the current model
    feature_importance[f"Model {i+1}"] = feature_imp

# Create a figure and axis for plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Customize colors for bars
bar_colors = 'blue'

# Plot feature importance for Model 1 (top 10 features)
top_10_features_1 = feature_importance["Model 1"]
axes[0].barh(top_10_features_1['Feature'], top_10_features_1['Value'], color=bar_colors)
axes[0].set_xlabel('Feature Importance')
axes[0].set_title('Top 10 Feature Importance for Model 1')

# Customize colors for bars
bar_colors = 'green'

# Plot feature importance for Model 2 (top 10 features)
top_10_features_2 = feature_importance["Model 2"]
axes[1].barh(top_10_features_2['Feature'], top_10_features_2['Value'], color=bar_colors)
axes[1].set_xlabel('Feature Importance')
axes[1].set_title('Top 10 Feature Importance for Model 2')

# Customize colors for bars
bar_colors = 'orange'

# Plot feature importance for Model 3 (top 10 features)
top_10_features_3 = feature_importance["Model 3"]
axes[2].barh(top_10_features_3['Feature'], top_10_features_3['Value'], color=bar_colors)
axes[2].set_xlabel('Feature Importance')
axes[2].set_title('Top 10 Feature Importance for Model 3')

# Adjust layout
plt.tight_layout()

# Save the plot to the specified file path
file_path = '/Users/d18127085/Desktop/statistical_phaseII/src/results/tech_top_10_features_plot.png'
plt.savefig(file_path, dpi=300, bbox_inches='tight')





set_trace()
# Methods2 shap
explainer = shap.Explainer(final_clf)

# Show the plot (optional)
file_path_shap= '/Users/d18127085/Desktop/statistical_phaseII/src/results/tech10_shap.png'
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.show()

# import shap
# import shap
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Define a dictionary to store SHAP values for each model
# shap_values_dict = {}

# for i, final_clf in enumerate(final_classifiers):
#     # Create a SHAP explainer for the current model
#     explainer = shap.Explainer(final_clf)
    
#     # Calculate SHAP values for the validation set
#     shap_values = explainer.shap_values(X_valid)
    
#     # Ensure that shap_values is 2-dimensional (if it's a list of ndarrays)
#     if isinstance(shap_values, list):
#         shap_values = np.concatenate(shap_values, axis=0)
    
#     # Calculate the average absolute SHAP values for each feature
#     avg_shap_values = np.abs(shap_values).mean(axis=0)
    
#     # Ensure that avg_shap_values is 1-dimensional
#     avg_shap_values = avg_shap_values.ravel()
    
#     # Sort and select the top 10 features based on SHAP values
#     shap_features = pd.DataFrame({'Feature': X.columns, 'Value': avg_shap_values})
#     shap_features = shap_features.sort_values(by='Value', ascending=False)
    
    # # Store the top features and their SHAP values for the current model
    # shap_values_dict[f"Model {i+1}"] = shap_features
    # # Access SHAP values for a specific model (e.g., Model 1)
    # shap_values_model = shap_values_dict["Model 1"]

    # # Create a summary plot
    # shap.summary_plot(shap_values_model['Value'], X_valid, feature_names=shap_values_model['Feature'], show=False)

    # # Save the summary plot to a file
    # file_path_summary_plot = 'summary_plot_model_1.png'
    # plt.savefig(file_path_summary_plot, dpi=300, bbox_inches='tight')


# Summary Plot for Important Features:
# The summary plot displays the overall feature importance and the impact of each feature on model predictions.
# It's a great way to visualize which features have the most significant effect on the model's output.

# # Access SHAP values for a specific model (e.g., Model 1)
# shap_values_model = shap_values_dict["Model 1"]

# # Create a summary plot
# shap.summary_plot(shap_values_model['Value'], X_valid, feature_names=shap_values_model['Feature'], show=False)

# # Save the summary plot to a file
# file_path_summary_plot = 'summary_plot_model_1.png'
# plt.savefig(file_path_summary_plot, dpi=300, bbox_inches='tight')


# # Define a dictionary to store SHAP values for each model
# shap_values_dict = {}

# for i, final_clf in enumerate(final_classifiers):
#     # Create a SHAP explainer for the current model
#     explainer = shap.Explainer(final_clf)
    
#     # Calculate SHAP values for the validation set
#     shap_values = explainer.shap_values(X_valid)
    
#     # Ensure that shap_values is 2-dimensional (if it's a list of ndarrays)
#     if isinstance(shap_values, list):
#         shap_values = np.concatenate(shap_values, axis=0)
    
#     # Calculate the average absolute SHAP values for each feature
#     avg_shap_values = np.abs(shap_values).mean(axis=0)
    
#     # Ensure that avg_shap_values is 1-dimensional
#     avg_shap_values = avg_shap_values.ravel()
    
#     # Sort and select the top 10 features based on SHAP values
#     shap_features = pd.DataFrame({'Feature': X.columns, 'Value': avg_shap_values})
#     shap_features = shap_features.sort_values(by='Value', ascending=False)
    
#     # Store the top features and their SHAP values for the current model
#     shap_values_dict[f"Model {i+1}"] = shap_features
#     # Access SHAP values for Model 1
#     shap_values_model_1 = shap_values_dict["Model 1"]

#     # Print or use the SHAP values as needed
#     print("SHAP values for Model 1:")
#     print(shap_values_model_1)

#     # Access SHAP values for Model 2
#     shap_values_model_2 = shap_values_dict["Model 2"]

#     # Print or use the SHAP values as needed
#     print("SHAP values for Model 2:")
#     print(shap_values_model_2)

#     # Access SHAP values for Model 3 (corrected variable name)
#     shap_values_model_3 = shap_values_dict["Model 3"]

#     # Print or use the SHAP values as needed
#     print("SHAP values for Model 3:")
#     print(shap_values_model_3)


# Summary Plot for Important Features:
# The summary plot displays the overall feature importance and the impact of each feature on model predictions.
# It's a great way to visualize which features have the most significant effect on the model's output.

   


# Dependence Plot for Variable Relationships:
# Dependence plots allow you to visualize the relationship between a specific feature and the model's output.
# You can create multiple dependence plots to explore the relationships between different features and the target variable.

# Select a feature you want to visualize (e.g., 'feature_name')
# feature_to_visualize = 'feature_name'

# # Create a dependence plot for the selected feature
# shap.dependence_plot(feature_to_visualize, shap_values, X_valid, feature_names=X.columns, show=False)

# # Save the dependence plot to a file
# file_path_dependence_plot = 'dependence_plot.png'
# plt.savefig(file_path_dependence_plot, dpi=300, bbox_inches='tight')

# Select a feature you want to visualize (e.g., 'age' or any other feature name)
# feature_to_visualize = 'age'

# # Create a dependence plot for the selected feature
# shap.dependence_plot(feature_to_visualize, shap_values, X_valid, feature_names=X.columns, show=False)

# # Save the dependence plot to a file
# file_path_dependence_plot = 'dependence_plot.png'
# plt.savefig(file_path_dependence_plot, dpi=300, bbox_inches='tight')



set_trace()
# import shap

# # Define a dictionary to store SHAP values for each model
# shap_values_dict = {}

# for i, final_clf in enumerate(final_classifiers):
#     # Create a SHAP explainer for the current model
#     explainer = shap.Explainer(final_clf)
    
#     # Calculate SHAP values for the validation set
#     shap_values = explainer.shap_values(X_valid)
    
#     # Ensure that shap_values is 2-dimensional (if it's a list of ndarrays)
#     if isinstance(shap_values, list):
#         shap_values = np.concatenate(shap_values, axis=0)
    
#     # Calculate the average absolute SHAP values for each feature
#     avg_shap_values = np.abs(shap_values).mean(axis=0)
    
#     # Ensure that avg_shap_values is 1-dimensional
#     avg_shap_values = avg_shap_values.ravel()
    
#     # Sort and select the top 10 features based on SHAP values
#     shap_features = pd.DataFrame({'Feature': X.columns, 'Value': avg_shap_values})
#     shap_features = shap_features.sort_values(by='Value', ascending=False)
    
#     # Store the top 10 features and their SHAP values for the current model
#     shap_values_dict[f"Model {i+1}"] = shap_features

    

# # Access SHAP values for Model 1, 
# shap_values_model_1 = shap_values_dict["Model 1"]

# # Print or use the SHAP values as needed
# print(shap_values_model_1)











# Now, final_classifiers contains the trained final classifiers for each target,
# and roc_auc_scores contains the ROC AUC scores for each target.



# # Define the number of splits and create StratifiedKFold
# n_splits = 10
# skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# final_classifiers = []

# # Define the data splits for each target (X_train1, y_train1, X_valid1, y_valid1, etc.)
# data_splits = [
#     (X_train1, y_train1, X_valid1, y_valid1),
#     (X_train2, y_train2, X_valid2, y_valid2),
#     (X_train3, y_train3, X_valid3, y_valid3)
# ]

# for i, (X_train, y_train, X_valid, y_valid) in enumerate(data_splits):
#     # Create an Optuna study and optimize the LightGBM model for the current target
#     study = optuna.create_study(direction='maximize')
#     study.optimize(lambda trial: lgbm_objective(trial, X_train, y_train, X_valid, y_valid), n_trials=1)
    
#     # Get the best parameters for the current target and create the final LightGBM model
#     best_params = study.best_params
#     final_clf = lgb.LGBMClassifier(**best_params, n_estimators=1000)
    
#     # Fit the final model to the entire training dataset for the current target
#     final_clf.fit(X_train, y_train)
    
#     final_classifiers.append(final_clf)

# # Evaluate the final models on the hold-out validation sets and make predictions for each target
# y_pred_vals = []

# for i, (X_train, y_train, X_valid, y_valid) in enumerate(data_splits):
#     final_clf = final_classifiers[i]
    
#     # Make predictions for the current target's validation set
#     y_pred_val = final_clf.predict_proba(X_valid)[:, 1]
#     y_pred_vals.append(y_pred_val)
#     set_trace()

#     # Fit the final models to the entire training datasets
#     final_clf1.fit(X_train1, y_train1)
#     final_clf2.fit(X_train2, y_train2)
#     final_clf3.fit(X_train3, y_train3)

#     # Evaluate the final models on the hold-out validation sets
#     y_pred_val1 = final_clf1.predict_proba(X_val1)[:, 1]
#     y_pred_val2 = final_clf2.predict_proba(X_val2)[:, 1]
#     y_pred_val3 = final_clf3.predict_proba(X_val3)[:, 1]

#     # Calculate ROC AUC scores on the validation sets for each target
#     roc_auc_val1 = round(roc_auc_score(y_val1, y_pred_val1), 3)
#     roc_auc_val2 = round(roc_auc_score(y_val2, y_pred_val2), 3)
#     roc_auc_val3 = round(roc_auc_score(y_val3, y_pred_val3), 3)
#     print('-----------')

# set_trace()


# # Define the objective function for Optuna optimization
# def lgbm_objective(trial, X_train, y_train, X_valid, y_valid):
#     params = {
#         'objective': 'binary',
#         'boosting_type': 'dart',
#         'num_leaves': trial.suggest_int('num_leaves', 2, 1000),
#         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 1000),
#         'max_depth': trial.suggest_int('max_depth', 1, 1000),
#         'subsample': trial.suggest_float('subsample', 0.1, 0.9),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.9),
#         'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.99),
#         'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1),
#         'n_estimators': 2000,
#         'random_state': 77,
#         'class_weight': 'balanced',
#     }

#     clf = lgb.LGBMClassifier(**params, early_stopping_rounds=50, verbose_eval=False)
#     clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='auc')
#     y_pred_valid = clf.predict_proba(X_valid)[:, 1]
#     roc_auc = roc_auc_score(y_valid, y_pred_valid)

#     return roc_auc

# # Define the number of splits and create StratifiedKFold
# n_splits = 10
# skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# final_classifiers = []

# # Define the data splits for each target (X_train1, y_train1, X_valid1, y_valid1, etc.)
# data_splits = [
#     (X_train1, y_train1, X_valid1, y_valid1),
#     (X_train2, y_train2, X_valid2, y_valid2),
#     (X_train3, y_train3, X_valid3, y_valid3)
# ]
# for i, (X_train, y_train, X_valid, y_valid) in enumerate(data_splits):
#     # Create an Optuna study and optimize the LightGBM model for the current target
#     study = optuna.create_study(direction='maximize')
#     study.optimize(lambda trial: lgbm_objective(trial, X_train, y_train, X_valid, y_valid), n_trials=1)
    
#     # Get the best parameters for the current target and create the final LightGBM model
#     best_params = study.best_params
#     final_clf = lgb.LGBMClassifier(**best_params, n_estimators=1000)
    
#     # Fit the final model to the entire training dataset for the current target
#     final_clf.fit(X_train, y_train)
    
#     final_classifiers.append(final_clf)

# # Evaluate the final models on the hold-out validation sets and make predictions for each target
# y_pred_vals = []

# for i, (X_train, y_train, X_valid, y_valid) in enumerate(data_splits):
#     final_clf = final_classifiers[i]
    
#     # Make predictions for the current target's validation set
#     y_pred_val = final_clf.predict_proba(X_valid)[:, 1]
#     y_pred_vals.append(y_pred_val)

#     # Fit the final models to the entire training datasets
#     final_clf1.fit(X_train1, y_train1)
#     final_clf2.fit(X_train2, y_train2)
#     final_clf3.fit(X_train3, y_train3)


#     # Evaluate the final models on the hold-out validation sets
#     y_pred_val1 = final_clf1.predict_proba(X_val1)[:, 1]
#     y_pred_val2 = final_clf2.predict_proba(X_val2)[:, 1]
#     y_pred_val3 = final_clf3.predict_proba(X_val3)[:, 1]


#     # Calculate ROC AUC scores on the validation sets for each target
#     roc_auc_val1 = round(roc_auc_score(y_val1, y_pred_val1), 3)
#     roc_auc_val2 = round(roc_auc_score(y_val2, y_pred_val2), 3)
#     roc_auc_val3 = round(roc_auc_score(y_val3, y_pred_val3), 3)
#     print('-----------')

# Now, y_pred_vals contains the predictions for each target's validation set.


# Now, you have a list of trained final classifiers (final_classifiers) for each fold.


set_trace()




# # Define the objective function for Optuna optimization
# def lgbm_objective(trial, X_train, y_train, X_valid, y_valid):
#     params = {
#         'objective': 'binary',
#         'boosting_type': 'dart',
#         'num_leaves': trial.suggest_int('num_leaves', 2, 1000),
#         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 1000),
#         'max_depth': trial.suggest_int('max_depth', 1, 1000),
#         'subsample': trial.suggest_float('subsample', 0.1, 0.9),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.9),
#         'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.99),
#         'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.5),
#         'n_estimators': 2000,
#         'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
#         'random_state': 77,
#         'class_weight': 'balanced',
          
#     }

#     clf = lgb.LGBMClassifier(**params, early_stopping_rounds=50, verbose_eval=False)
#     clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='auc')
#     y_pred_valid = clf.predict_proba(X_valid)[:, 1]
#     roc_auc = roc_auc_score(y_valid, y_pred_valid)

#     return roc_auc

# # Create an Optuna study and optimize the LightGBM model for each target separately
# study1 = optuna.create_study(direction='maximize')
# study1.optimize(lambda trial: lgbm_objective(trial, X_train1, y_train1, X_valid1, y_valid1), n_trials=1)

# study2 = optuna.create_study(direction='maximize')
# study2.optimize(lambda trial: lgbm_objective(trial, X_train2, y_train2, X_valid2, y_valid2), n_trials=1)

# study3 = optuna.create_study(direction='maximize')
# study3.optimize(lambda trial: lgbm_objective(trial, X_train3, y_train3, X_valid3, y_valid3), n_trials=1)

# # Get the best parameters for each target and create the final LightGBM models
# best_params1 = study1.best_params
# final_clf1 = lgb.LGBMClassifier(**best_params1, n_estimators=1000)

# best_params2 = study2.best_params
# final_clf2 = lgb.LGBMClassifier(**best_params2, n_estimators=1000)

# best_params3 = study3.best_params
# final_clf3 = lgb.LGBMClassifier(**best_params3, n_estimators=1000)

# Fit the final models to the entire training datasets
final_clf1.fit(X_train1, y_train1)
final_clf2.fit(X_train2, y_train2)
final_clf3.fit(X_train3, y_train3)


# Evaluate the final models on the hold-out validation sets
y_pred_val1 = final_clf1.predict_proba(X_val1)[:, 1]
y_pred_val2 = final_clf2.predict_proba(X_val2)[:, 1]
y_pred_val3 = final_clf3.predict_proba(X_val3)[:, 1]


# Calculate ROC AUC scores on the validation sets for each target
roc_auc_val1 = round(roc_auc_score(y_val1, y_pred_val1), 3)
roc_auc_val2 = round(roc_auc_score(y_val2, y_pred_val2), 3)
roc_auc_val3 = round(roc_auc_score(y_val3, y_pred_val3), 3)
print('-----------')
# Print where each study stopped
print("Study 1 stopped at trial:", len(study1.trials))
print("Study 2 stopped at trial:", len(study2.trials))
print("Study 3 stopped at trial:", len(study3.trials))

print('-----------')

print('ROC AUC on the validation set for tech10-d1:', roc_auc_val1)

print('ROC AUC on the validation set for tech10-d2:', roc_auc_val2)

print('ROC AUC on the validation set for tech10-d3:', roc_auc_val3)

print('-----------')




# Method 1: LightGBM Feature Importance
feature_imp1 = pd.DataFrame(sorted(zip(final_clf1.feature_importances_, X.columns)), columns=['Value', 'Feature'])
feature_imp2 = pd.DataFrame(sorted(zip(final_clf2.feature_importances_, X.columns)), columns=['Value', 'Feature'])
feature_imp3 = pd.DataFrame(sorted(zip(final_clf3.feature_importances_, X.columns)), columns=['Value', 'Feature'])

# Select the top 10 features for each model

top_10_features_1 = feature_imp1.nlargest(10, 'Value').reset_index(drop=True)
top_10_features_2 = feature_imp2.nlargest(10, 'Value').reset_index(drop=True)
top_10_features_3 = feature_imp3.nlargest(10, 'Value').reset_index(drop=True)





# top_features_list = feature_imp1['Feature'].head(10).tolist()  # Get the top 10 feature names as a list
# top_10_features_1 = feature_imp1.nlargest(10, 'Value').reset_index(drop=True)

# # Get the top 10 feature names as a list model 1
# top_features_list = feature_imp1['Feature'].head(10).tolist()

# # Create a DataFrame containing the top 10 features based on their values
# top_10_features_1 = feature_imp1.nlargest(10, 'Value').reset_index(drop=True)

# # Get the top 10 feature names as a list model 2
# top_features_list2 = feature_imp2['Feature'].head(10).tolist()

# # Create a DataFrame containing the top 10 features based on their values
# top_10_features_2 = feature_imp2.nlargest(10, 'Value').reset_index(drop=True)


# # Get the top 10 feature names as a list model 3
# top_features_list3 = feature_imp3['Feature'].head(10).tolist()

# # Create a DataFrame containing the top 10 features based on their values
# top_10_features_3 = feature_imp3.nlargest(10, 'Value').reset_index(drop=True)

# top_10_features_1 = feature_imp1.nlargest(10, 'Value')
# top_10_features_2 = feature_imp2.nlargest(10, 'Value')
# top_10_features_3 = feature_imp3.nlargest(10, 'Value')



print('+++++++++++++++++++++++++++++++++++')


print(type(y_val1))
print(type(y_pred_val1))


y_val1 = y_val1.astype(int)
y_pred_val1 = y_pred_val1.astype(int)

y_val2 = y_val2.astype(int)
y_pred_val2 = y_pred_val2.astype(int)

y_val3 = y_val3.astype(int)
y_pred_val3 = y_pred_val1.astype(int)




# Metrics Dictionary for each final_clf1, final_clf2, final_clf3
metrics_dict1 = {}
cm1 = confusion_matrix(y_val1, y_pred_val1)
Recall1 = cm1[1][1] / (cm1[1][1] + cm1[1][0])
specificity1 = cm1[0][0] / (cm1[0][0] + cm1[0][1])
f1_1 = f1_score(y_val1, y_pred_val1, average='macro')
f1w_1 = f1_score(y_val1, y_pred_val1, average='weighted')
bas_1 = balanced_accuracy_score(y_val1, y_pred_val1)
ps_1 = precision_score(y_val1, y_pred_val1, average='weighted')

metrics_dict1 = {
    'Classifier': 'LGBM',
    'Target_Value': 'tech10-d1',
    'TP': cm1[1][1],
    'FP': cm1[0][1],
    'TN': cm1[0][0],
    'FN': cm1[1][0],
    'Recall': Recall1,
    'Specificity': specificity1,
    'Precision': ps_1,
    'Balanced_Accuracy': bas_1,
    'F1_Weighted': f1w_1,
    'roc_auc_val' :roc_auc_val1
   
    # 'Feature_Importance': feature_imp1,
    # 'RFE': selected_features1,
    # 'SFS': selected_features1
}

metrics_dict2 = {}
cm2 = confusion_matrix(y_val2, y_pred_val2)
Recall2 = cm2[1][1] / (cm2[1][1] + cm2[1][0])
specificity2 = cm2[0][0] / (cm2[0][0] + cm2[0][1])
f1_2 = f1_score(y_val2, y_pred_val2, average='macro')
f1w_2 = f1_score(y_val2, y_pred_val2, average='weighted')
bas_2 = balanced_accuracy_score(y_val2, y_pred_val2)
ps_2 = precision_score(y_val2, y_pred_val2, average='weighted')

metrics_dict2 = {
    'Classifier': 'LGBM',
    'Target_Value': 'tech10-d2',
    'TP': cm2[1][1],
    'FP': cm2[0][1],
    'TN': cm2[0][0],
    'FN': cm2[1][0],
    'Recall': Recall2,
    'Specificity': specificity2,
    'Precision': ps_2,
    'Balanced_Accuracy': bas_2,
    'F1_Weighted': f1w_2,
     'roc_auc_val' :roc_auc_val2
    # 'Feature_Importance': feature_imp2,
    # 'RFE': selected_features2,
    # 'SFS': selected_features2
}

metrics_dict3 = {}
cm3 = confusion_matrix(y_val3, y_pred_val3)
Recall3 = cm3[1][1] / (cm3[1][1] + cm3[1][0])
specificity3 = cm3[0][0] / (cm3[0][0] + cm3[0][1])
f1_3 = f1_score(y_val3, y_pred_val3, average='macro')
f1w_3 = f1_score(y_val3, y_pred_val3, average='weighted')
bas_3 = balanced_accuracy_score(y_val3, y_pred_val3)
ps_3 = precision_score(y_val3, y_pred_val3, average='weighted')

metrics_dict3 = {
    'Classifier': 'LGBM',
    'Target_Value': 'tech10-d3',
    'TP': cm3[1][1],
    'FP': cm3[0][1],
    'TN': cm3[0][0],
    'FN': cm3[1][0],
    'Recall': Recall3,
    'Specificity': specificity3,
    'Precision': ps_3,
    'Balanced_Accuracy': bas_3,
    'F1_Weighted': f1w_3,
    'roc_auc_val' :roc_auc_val3
    # 'Feature_Importance': feature_imp3,
    # 'RFE': selected_features3,
    # 'SFS': selected_features3
}



# Create DataFrames from the metrics dictionaries
df1 = pd.DataFrame(metrics_dict1, index=[0])
df2 = pd.DataFrame(metrics_dict2, index=[0])
df3 = pd.DataFrame(metrics_dict3, index=[0])

# Concatenate the DataFrames vertically to create a single DataFrame
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# Display the combined DataFrame
print(combined_df.T)
combined_df.to_csv('/Users/d18127085/Desktop/statistical_phaseII/src/results/tech10_combined_data.csv', index=False)



# Plotting the metrics


def plot_confusion_matrix(cm, model_name, ax):
    ax = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False,
                     annot_kws={"size": 14}, fmt='g', ax=ax)
    cmlabels = ['True Negatives', 'False Positives',
                'False Negatives', 'True Positives']
    for i, t in enumerate(ax.texts):
        t.set_text(t.get_text() + "\n" + cmlabels[i])
    ax.set_title(f'Confusion Matrix for {model_name}', size=15)
    ax.set_xlabel('Predicted Values', size=13)
    ax.set_ylabel('True Values', size=13)

# Create subplots for all three confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Assuming cm1, cm2, and cm3 are your confusion matrices for Model 1, Model 2, and Model 3
plot_confusion_matrix(cm1, 'Model 1', axes[0])
plot_confusion_matrix(cm2, 'Model 2', axes[1])
plot_confusion_matrix(cm3, 'Model 3', axes[2])

# plt.show()

file_path_cm = '/Users/d18127085/Desktop/statistical_phaseII/src/results/tech10_CM.png'
# Save the plot to the specified file path
plt.savefig(file_path_cm, dpi=300, bbox_inches='tight')
# Close the plot 
plt.close()




# Filter the DataFrame to select rows with the desired metrics (precision, Recall, balanced accuracy, and F1 weighted)
selected_metrics = ['Precision', 'Recall', 'Balanced_Accuracy', 'F1_Weighted','roc_auc_val']
filtered_df = combined_df[combined_df['Classifier'].isin(['LGBM'])]  # Filter by Classifier if needed

# Create a dictionary to store the results for each model
results = {}

# Loop through the models and store the selected metrics in the results dictionary
for model_name in ['tech10-d1', 'tech10-d2', 'tech10-d3']:
    model_data = filtered_df[filtered_df['Target_Value'] == model_name]
    results[model_name] = model_data[selected_metrics].values[0]

# Create a bar plot
plt.figure(figsize=(12, 5))

bar_width = 0.2
index = np.arange(len(selected_metrics))

for i, (model_name, metric_values) in enumerate(results.items()):
    plt.bar(index + i * bar_width, metric_values, width=bar_width, label=model_name)

plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Metrics Comparison Across Persuasive Tasks and Models')
plt.xticks(index + bar_width, selected_metrics)
plt.legend()
# plt.show()
# Save the plot to an image file 
# plt.savefig('comparison_plot_tech10.png', dpi=300, bbox_inches='tight') 
file_path_metrics = '/Users/d18127085/Desktop/statistical_phaseII/src/results/tech10_comparison_plot.png'
# Save the plot to the specified file path
plt.savefig(file_path_metrics, dpi=300, bbox_inches='tight')
# Close the plot 
plt.close()





# Create a figure and axis
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Customize colors for bars
bar_colors = 'blue'

# Plot feature importance for Model 1 (top 10 features)
axes[0].barh(top_10_features_1['Feature'], top_10_features_1['Value'], color=bar_colors)
axes[0].set_xlabel('Feature Importance')
axes[0].set_title('Top 10 Feature Importance for Model 1')

# Customize colors for bars
bar_colors = 'green'

# Plot feature importance for Model 2 (top 10 features)
axes[1].barh(top_10_features_2['Feature'], top_10_features_2['Value'], color=bar_colors)
axes[1].set_xlabel('Feature Importance')
axes[1].set_title('Top 10 Feature Importance for Model 2')

# Customize colors for bars
bar_colors = 'orange'

# Plot feature importance for Model 3 (top 10 features)
axes[2].barh(top_10_features_3['Feature'], top_10_features_3['Value'], color=bar_colors)
axes[2].set_xlabel('Feature Importance')
axes[2].set_title('Top 10 Feature Importance for Model 3')

# Adjust layout
plt.tight_layout()

# Save the plot to the specified file path
file_path = '/Users/d18127085/Desktop/statistical_phaseII/src/results/tech10_features_plot.png'
plt.savefig(file_path, dpi=300, bbox_inches='tight')

# Show the plot (optional)
# plt.show()





# Compute ROC curve and ROC AUC for each classifier

# Model 1
fpr1, tpr1, _ = roc_curve(y_val1, y_pred_val1)
roc_auc1 = roc_auc_score(y_val1, y_pred_val1)

# Model 2
fpr2, tpr2, _ = roc_curve(y_val2, y_pred_val2)
roc_auc2 = roc_auc_score(y_val2, y_pred_val2)

# Model 3
fpr3, tpr3, _ = roc_curve(y_val3, y_pred_val3)
roc_auc3 = roc_auc_score(y_val3, y_pred_val3)

# Create a single plot for ROC Curves
plt.figure(figsize=(8, 6))

# Plot ROC curves for each classifier
plt.plot(fpr1, tpr1, label=f"Model 1 (AUC = {roc_auc1:.2f})")
plt.plot(fpr2, tpr2, label=f"Model 2 (AUC = {roc_auc2:.2f})")
plt.plot(fpr3, tpr3, label=f"Model 3 (AUC = {roc_auc3:.2f})")

# Plot the no-skill line
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')

# Set plot limits and labels
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")

# Add a legend to the plot
plt.legend(loc="lower right")

# Adjust the layout for better presentation
plt.tight_layout()

# Specify the file path for saving the plot
file_path_auc = '/Users/d18127085/Desktop/statistical_phaseII/src/results/tech10_aucs_plot.png'

# Save the plot as an image with high DPI and tight bounding box
plt.savefig(file_path_auc, dpi=300, bbox_inches='tight')





































# # Compute ROC curve and ROC AUC for each classifier
# fpr1, tpr1, _ = roc_curve(y_val1, y_pred_val1)
# roc_auc1 = roc_auc_score(y_val1, y_pred_val1)

# fpr2, tpr2, _ = roc_curve(y_val2, y_pred_val2)
# roc_auc2 = roc_auc_score(y_val2, y_pred_val2)

# fpr3, tpr3, _ = roc_curve(y_val3, y_pred_val3)
# roc_auc3 = roc_auc_score(y_val3, y_pred_val3)

# # Create a single plot for ROC Curves
# plt.figure(figsize=(8, 6))

# # Plot ROC curves for each classifier
# plt.plot(fpr1, tpr1, label=f"Model 1 (AUC = {roc_auc1:.2f})")
# plt.plot(fpr2, tpr2, label=f"Model 2 (AUC = {roc_auc2:.2f})")
# plt.plot(fpr3, tpr3, label=f"Model 3 (AUC = {roc_auc3:.2f})")

# # Plot the no-skill line
# plt.plot([0, 1], [0, 1], linestyle='**', label='No Skill')

# # Set plot limits and labels
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend(loc="lower right")

# plt.tight_layout()
# file_path_auc= '/Users/d18127085/Desktop/statistical_phaseII/src/results/tech10_aucs_plot.png'
# plt.savefig(file_path_auc, dpi=300, bbox_inches='tight')







# # # Compute ROC curve and ROC AUC for each classifier
# fpr1, tpr1, _ = roc_curve(y_val1, y_pred_val1)
# roc_auc1 = roc_auc_score(y_val1, y_pred_val1)

# fpr2, tpr2, _ = roc_curve(y_val2, y_pred_val2)
# roc_auc2 = roc_auc_score(y_val2, y_pred_val2)

# fpr3, tpr3, _ = roc_curve(y_val3, y_pred_val3)
# roc_auc3 = roc_auc_score(y_val3, y_pred_val3)

# # Create a single plot for ROC Curves
# plt.figure(figsize=(8, 6))

# # Plot ROC curves for each classifier
# plt.plot(fpr1, tpr1, label=f"Model 1 (AUC = {roc_auc1:.2f})")
# plt.plot(fpr2, tpr2, label=f"Model 2 (AUC = {roc_auc2:.2f})")
# plt.plot(fpr3, tpr3, label=f"Model 3 (AUC = {roc_auc3:.2f})")



# # Plot the no-skill line
# plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')

# # Set plot limits and labels
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend(loc="lower right")

# plt.tight_layout()
# file_path_auc2 = '/Users/d18127085/Desktop/statistical_phaseII/src/results/tech10_aucs_plot.png'
# plt.savefig(file_path_auc2, dpi=300, bbox_inches='tight')



set_trace()