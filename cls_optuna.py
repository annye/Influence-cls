
from common_imports import *



def optimize_and_evaluate_lgbm(X, y, X_train, y_train):


    # Step 1: Define the objective function for Optuna
    def objective(trial):
        # Define hyperparameters to optimize
        # ...
        params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": "auc",
        "n_jobs": -1,
        "verbose": -1,
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-6, 20.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-6, 20.0),
        "num_leaves": trial.suggest_int("num_leaves", 2, 200), 
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.0001, 0.5),  
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.1, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 1000),  
        "subsample": trial.suggest_uniform("subsample", 0.1, 1.5),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 2.0),
        "max_depth": trial.suggest_categorical("max_depth", [3, 6, 9, 12, 15, 18, 21, 24, 30, 60, 90]), 
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 1e-6, 100.0),  # Increase the upper bound
        "min_split_gain": trial.suggest_loguniform("min_split_gain", 1e-6, 1000),
        }

        inner_kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        inner_auc_scores = []

        for train_idx, val_idx in inner_kf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)

            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

            # Specify early stopping criteria
        
            valid_sets = [train_data, val_data]

      
            # Specify early stopping criteria
            params['early_stopping_round'] = 100
            params['verbose'] = 100  # Adjust the verbosity level if needed

            # Train the model with early stopping using train method
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000000,  
                valid_sets=[train_data, val_data],
            )

            # y_pred = model.predict_proba(X_val_fold)[:, 1]
            y_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)

            auc = roc_auc_score(y_val_fold, y_pred)
            inner_auc_scores.append(auc)


        inner_mean_auc = np.mean(inner_auc_scores)
        return inner_mean_auc

    # Step 2: Split data into training and test sets (for the outer cross-validation)
    X_train_outer, X_test_outer, y_train_outer, y_test_outer = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    #################################Trials###############################
    # Step 3: Create an outer Optuna study for hyperparameter tuning
    outer_study = optuna.create_study(direction="maximize")
    outer_study.optimize(objective, n_trials=100)  # adjust the number of trials

    # Step 4: Retrieve the best hyperparameters from the outer study
    best_params = outer_study.best_params

    # Step 5: Train a final model using the best hyperparameters on the entire training dataset
    # final_model = lgb.LGBMClassifier(**best_params)
    final_model = lgb.LGBMClassifier(**best_params, class_weight='balanced')
    final_model.fit(X_train_outer, y_train_outer)

    # Step 6: Evaluate the final model on the hold-out test set (from the outer cross-validation)
    y_pred_outer = final_model.predict_proba(X_test_outer)[:, 1]

    # Adjust the threshold 
    threshold = 0.5
    y_pred_outer_adjusted = (y_pred_outer > threshold).astype(int)

    auc_outer = roc_auc_score(y_test_outer, y_pred_outer)
    auc_outer_adjusted = roc_auc_score(y_test_outer, y_pred_outer_adjusted)

    # Print AUC scores for both original and adjusted thresholds
    print(f"AUC on the test set (outer fold): {auc_outer}")
    print(f"AUC on the test set (outer fold, adjusted threshold): {auc_outer_adjusted}")

    # Step 8: Get the confusion matrix for the final model with the adjusted threshold
    cm = confusion_matrix(y_test_outer, y_pred_outer_adjusted)
    print("Confusion Matrix (Adjusted Threshold):")
    print(cm)
    # Calculate metrics
    TP = cm[1][1]
    FP = cm[0][1]
    TN = cm[0][0]
    FN = cm[1][0]
    Recall = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    Precision = TP / (TP + FP)
    Balanced_Accuracy = (Recall + Specificity) / 2

   

    ############ Feature Importances#################


    # Extract feature importances
    feature_importance = final_model.feature_importances_

    # Combine feature names and their importances into a list of tuples
    feature_importance_list = list(zip(X_train_outer.columns, feature_importance))

    # Sort the list based on feature importances in descending order
    sorted_feature_importance = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)

    # Extract the top ten features
    top_ten_features = sorted_feature_importance[:10]

    # Print the top ten features
    for feature, importance in top_ten_features:
        print(f"Feature: {feature}, Importance: {importance}")

    # Customize colors for bars
    bar_colors = 'blue'

    # Create a list of feature names and their importances for plotting
    top_10_features_1 = pd.DataFrame(top_ten_features, columns=['Feature', 'Value'])

    # Create subplots
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))

    # Create a horizontal bar plot
    axes.barh(top_10_features_1['Feature'], top_10_features_1['Value'], color=bar_colors)
    axes.set_xlabel('Feature Importance')
    axes.set_title('Top 10 Feature Importance for Model X')

   # plt.show()

###########################################################
# Calculate average absolute SHAP values for each feature
  

    # Create a SHAP explainer for your model
    explainer = shap.TreeExplainer(final_model)  # Replace 'final_model' with your trained model
    # Calculate SHAP values for your test set
    shap_values = explainer.shap_values(X_test_outer)  # Replace 'X_test_outer' with your test data




    avg_shap_values = np.abs(shap_values).mean(axis=0)

    # Convert the SHAP values to scalar representations for sorting
    shap_feature_importance = [(feature, value.tolist()) for feature, value in zip(X.columns, avg_shap_values)]

    # Sort the list based on SHAP values in descending order
    sorted_shap_feature_importance = sorted(shap_feature_importance, key=lambda x: x[1], reverse=True)

    # Extract the top ten features
    top_ten_shap_features = sorted_shap_feature_importance[:10]

    # Plot the top 10 features
    top_10_feature_names, top_10_shap_values = zip(*top_ten_shap_features)
    

    # Customize colors for bars
    bar_colors = 'blue'

    # Create a list of feature names and their SHAP values for plotting
    top_10_shap_features_df = pd.DataFrame(top_ten_shap_features, columns=['Feature', 'SHAP Value'])

    # # Create subplots
    # fig, axes = plt.subplots(1, 1, figsize=(10, 6))

    # # Create a horizontal bar plot for the top 10 feature  s
    # axes.barh(top_10_feature_names, top_10_shap_values, color=bar_colors)
    # axes.set_xlabel('SHAP Value')
    # axes.set_title('Top 10 SHAP Values for Model X')

    #plt.show()
    
    
#########################################################################################################
    metrics_dict = {
        "Classifier": "LGBM-optuna",
        "Target_Value": "tech10-d2",  # Replace with target value
        "TP": cm[1][1],
        "FP": cm[0][1],
        "TN": cm[0][0],
        "FN": cm[1][0],
        "Sensitivity": Recall,
        "Specificity": Specificity,
        "Precision": Precision,
        "Balanced Accuracy": Balanced_Accuracy,
        "Features": top_ten_features,
        "features_values": top_10_shap_features_df,
        "Shap Features": top_ten_shap_features,
        "shap_values":  shap_values
        }
   # Convert the dictionary to a DataFrame
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient="index"). T

    # Save the DataFrame to a CSV file
    results = metrics_df.to_csv(r'C:\persuasion\src\metrics.csv', index=False)

    set_trace()

        
    
    





        

   