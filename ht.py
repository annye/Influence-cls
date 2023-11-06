from common_imports import *

# X, df_techniques = get_data()  

class HyperoptTuning:

    def __init__(self, algo, X_train, X_test, y_train, y_test, hyperopt_rstate=42):
        self.algo = algo
        self.hyperopt_rstate = hyperopt_rstate
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        # self.X_val = 
        # self.y_val = 
        
        self.obj_fns = {
            'LR': self.objective_lr,
            'SVM': self.objective_svm,
            'RF': self.objective_rf,
            'XGB': self.objective_xgb,
            'NB': self.objective_nb,
            'LGBM': self.objective_lgbm,
            'MLP': self.objective_mlp
        }
        
  
    param_dist_hyperopt = {
        'num_leaves': hp.choice('num_leaves', range(10, 100)),
        'learning_rate': hp.loguniform('learning_rate', -5, 0, 0.5),
        'max_depth': hp.choice('max_depth', range(1, 70)),
        'min_child_weight': hp.choice('min_child_weight', range(1, 70)),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': hp.uniform('reg_alpha', 0, 1),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        'n_estimators': hp.choice('n_estimators', range(50, 1000)),
    }

    # Search for MLP
    # param_dist_hyperopt = {
    #                 'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50, 100)]),
    #                 'activation': hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']),
    #                 'solver': hp.choice('solver', ['lbfgs', 'sgd', 'adam']),
    #                 'alpha': hp.loguniform('alpha', -7, 0),  # Alpha (L2 regularization) in log scale
    #                 'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']),
    #                 'learning_rate_init': hp.loguniform('learning_rate_init', -5, 0),
    #             }


                    

    # Objective Functions
    def objective_lr(self, params):
        # Implement the objective function for Logistic Regression
        model = LogisticRegression(random_state=42, max_iter=100, n_jobs=-1, **params) #, max_iter=5000
        model.fit(self.X_train, self.y_train)  
        if len(set(self.y_train)) == 2:
            y_probas = model.predict_proba(self.X_test)[:, 1]
            roc_auc = roc_auc_score(self.y_test, y_probas)
        elif len(set(self.y_train)) > 2:
            y_probas = model.predict_proba(self.X_test)
            roc_auc = roc_auc_score(self.y_test, y_probas, multi_class='ovr', average='micro')  
        return -roc_auc

            
    def objective_svm(self, params):
        # Implement the objective function for SVM
        model = SVC(random_state=42, probability=True, **params)
        model.fit(self.X_train, self.y_train)
        if len(set(self.y_train)) == 2:
            y_probas = model.predict_proba(self.X_test)[:, 1]
            roc_auc = roc_auc_score(self.y_test, y_probas)
        elif len(set(self.y_train)) > 2:
            y_probas = model.predict_proba(self.X_test)
            roc_auc = roc_auc_score(self.y_test, y_probas, multi_class='ovr', average='micro')
        return -roc_auc
    
   
    def objective_nb(self, params):
        # Implement the objective function for Naive Bayes
        model = GaussianNB(**params)
        model.fit(self.X_train, self.y_train)
        if len(set(self.y_train)) == 2:
            y_probas = model.predict_proba(self.X_test)[:, 1]
            roc_auc = roc_auc_score(self.y_test, y_probas)
        elif len(set(self.y_train)) > 2:
            y_probas = model.predict_proba(self.X_test)
            roc_auc = roc_auc_score(self.y_test, y_probas, multi_class='ovr', average='micro')
        return -roc_auc

       
   
    def objective_rf(self, params):
        # Implement the objective function for Random Forest
        params['n_estimators'] = int(params['n_estimators'])
        model = RandomForestClassifier(random_state=42, **params)
        model.fit(self.X_train, self.y_train)
        if len(set(self.y_train)) == 2:
            y_probas = model.predict_proba(self.X_test)[:, 1]
            roc_auc = roc_auc_score(self.y_test, y_probas)
        elif len(set(self.y_train)) > 2:
            y_probas = model.predict_proba(self.X_test)
            roc_auc = roc_auc_score(self.y_test, y_probas, multi_class='ovr', average='micro')
        return -roc_auc


   
    def objective_xgb(self, params):
        # Implement the objective function for XGBoost
        model = XGBClassifier(random_state=42, **params)
        model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)],
                eval_metric='auc', verbose=False, early_stopping_rounds=10)
        if len(set(self.y_train)) == 2:
            y_probas = model.predict_proba(self.X_test)[:, 1]
            roc_auc = roc_auc_score(self.y_test, y_probas)
        elif len(set(self.y_train)) > 2:
            y_probas = model.predict_proba(self.X_test)
            roc_auc = roc_auc_score(self.y_test, y_probas, multi_class='ovr', average='micro')
        return -roc_auc



    def objective_lgbm(self, params):
        # Implement the objective function for LightGBM
        model = LGBMClassifier(random_state=42, **params)
        model.fit(self.X_train,
                self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                eval_metric='auc',
                verbose=False,
                early_stopping_rounds=10)

        if len(set(self.y_train)) == 2:
            y_probas = model.predict_proba(self.X_test)[:, 1]
            roc_auc = roc_auc_score(self.y_test, y_probas)
        elif len(set(self.y_train)) > 2:
            y_probas = model.predict_proba(self.X_test)
            roc_auc = roc_auc_score(self.y_test, y_probas, multi_class='ovr', average='micro')

        return -roc_auc
    
    def objective_mlp(self, params):
        # Implement the objective function for MLP Classifier
        model = MLPClassifier(random_state=42, **params)
        model.fit(self.X_train, self.y_train)
        if len(set(self.y_train)) == 2:
            y_probas = model.predict_proba(self.X_test)[:, 1]
            roc_auc = roc_auc_score(self.y_test, y_probas)
        elif len(set(self.y_train)) > 2:
            y_probas = model.predict_proba(self.X_test)
            roc_auc = roc_auc_score(self.y_test, y_probas, multi_class='ovr', average='micro')
        return -roc_auc
    

    def build_models(self, algos,threshold=0.5):
          # Logic to build models for selected algorithms
        algo = self.algo
        start = time.time()
        obj_fns = {'LR': self.objective_lr, 'SVM': self.objective_svm, 'RF': self.objective_rf,
                'XGB': self.objective_xgb, 'NB': self.objective_nb, 'LGBM': self.objective_lgbm}

        outer_predictions = {'Fold predictions': [], 'Fold probabilities': [], 'Fold test': []}
        cv_outer = StratifiedKFold(n_splits=5, shuffle=False)

        for train_ix, test_ix in cv_outer.split(self.X_train, self.y_train):
            # No need to reassign X_train, X_test, y_train, y_test; they are already available as instance variables

            if algo == 'XGB':
                X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)

            if algo == 'LGBM':
                X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)
            
            if algo == 'MLP':
                X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)


            if algo in ['LR', 'SVM', 'NB', 'MLP']:  # LGBM - Not scaled for RF or XGB
                # Scaling
                sc = StandardScaler()
                self.X_train = sc.fit_transform(self.X_train)
                self.X_test = sc.transform(self.X_test)

            space = return_parameter_space(algo)
            trials = Trials()

            # Create a RandomState object and pass it as rstate
            random_state = np.random.RandomState(self.hyperopt_rstate)
            best = fmin(fn=lambda params: obj_fns[algo](params),
                        space=space, algo=tpe.suggest, max_evals=100, trials=trials, rstate=random_state)

            # Retrieve the best parameters
            best_params = space_eval(space, best)
            if algo in ['LR']:
                best_model = LogisticRegression(random_state=42, n_jobs=-1, **best_params)
            elif algo == 'SVM':
                best_model = SVC(random_state=42, probability=True, **best_params)
            elif algo == 'RF':
                best_model = RandomForestClassifier(random_state=42, **best_params)
            elif algo == 'NB':
                best_model = GaussianNB(**best_params)
            elif algo == 'MLP':
                best_model == MLPClassifier(random_state=42, **best_params)
            elif algo == 'XGB':
                best_model = XGBClassifier(random_state=42, **best_params)
            elif algo == 'LGBM':
                best_model = LGBMClassifier(random_state=42, **best_params)

            best_model.fit(self.X_train, self.y_train)

            # evaluate model on the hold out dataset
            y_pred = best_model.predict(self.X_test)
            # Get predicted probabilities
            y_probas = best_model.predict_proba(self.X_test)[::, 1]
          
            # Apply threshold adjustment
            y_pred_adjusted = (y_probas >= threshold).astype(int)

            # Create a SHAP explainer for the best model
            explainer = shap.Explainer(best_model, self.X_train)

            # Compute SHAP values for the test data
            shap_values = explainer.shap_values(self.X_test)
            
            outer_predictions['Fold predictions'].append(y_pred_adjusted)
            outer_predictions['Fold probabilities'].append(y_probas)
            outer_predictions['Fold test'].append(self.y_test)

        # Summarize the estimated performance of the model over nested CV outer test sets
        results = get_and_record_scores(outer_predictions)
        save_results_dictionary(results, 'results_' + str(algo) + '_hyperopt.pkl')
        print("Duration for {}: {}".format(str(algo), time.time() - start))

        # Create a DataFrame from the results
        results_df = pd.DataFrame(results)

        # Define the Excel file path
        excel_file_path = 'results_' + str(algo) + '_hyperopt.xlsx'

        # Save the results to an Excel file
        results_df.to_excel(excel_file_path, index=False)

        print("Results saved to:", excel_file_path)
        print("Duration for {}: {}".format(str(algo), time.time() - start))



    
        #shap.summary_plot(shap_values, self.X_test)
        
        avg_shap_values = np.abs(shap_values).mean(axis=0)

        # Create a DataFrame with feature names and their corresponding SHAP values
        shap_features = pd.DataFrame({'Feature': X.columns, 'Value': avg_shap_values})

        # Sort the features based on their SHAP values in descending order
        shap_features = shap_features.sort_values(by='Value', ascending=False)

        # Select the top N features (e.g., top 10)
        top_n = 10
        top_features = shap_features.head(top_n)

        # Reverse the order to have the largest values at the top
        top_features = top_features.iloc[::-1]
       

        # Visualize the top features
        print(top_features)
        
       
      
        # # Create a bar plot
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.barh(top_features['Feature'], top_features['Value'], color='skyblue')
        plt.xlabel('SHAP Value')
        plt.ylabel('Feature')
        plt.title('Top Features based on SHAP Values')
        plt.grid(axis='x', linestyle='--', alpha=0.6)

        # Invert the y-axis to have the highest feature at the top
        ax.invert_yaxis()

        # Save the plot as an image file 
        plt.savefig('top_features_plot.png', bbox_inches='tight', dpi=300)

        # Show the plot
        plt.show()
        set_trace()

       

      

    
