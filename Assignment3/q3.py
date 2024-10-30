import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, fbeta_score
from q1 import data_preprocessing
from q2 import data_splits, normalize_features

# Step 1: Create hyperparameter grids for each model
# TODO fill out below dictionaries with reasonable values
param_grid_decision_tree = {
    'criterion':... ,
    'max_depth': ...,
    'min_samples_leaf': ...,
    'max_leaf_nodes': ...
}

param_grid_random_forest = {
    'n_estimators': ... ,
    'max_depth': ... ,
    'bootstrap': ... ,
}

param_grid_svm = {
    'kernel': ... ,
    'shrinking': ...,
    'C': ...,
    'tol': ...,
    'gamma': ...
}

# Step 2: Initialize classifiers with random_state=0
decision_tree = ... # TODO
random_forest = ... # TODO
svm = ... # TODO

# Step 3: Create a scorer using F-beta score with beta=0.5
scorer = ... # TODO


# Step 4: Perform grid search for each model using 9-fold StratifiedKFold cross-validation
def perform_grid_search(model, X_train, y_train, params):
    # Define the cross-validation strategy
    strat_kfold = ... # TODO

    # Grid search for the model
    grid_search = ... # TODO
    # TODO fit to the data

    best_param = ... # TODO
    best_score = ... # TODO
    print("Best parameters are:", best_param)
    print("Best score is:", best_score)

    # Return the fitted grid search objects
    return grid_search, best_param, best_score



X, y = data_preprocessing()
X_train, X_test, y_train, y_test = data_splits(X, y)
X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

# Do Grid search for Decision Tree
grid_decision_tree, best_params_decision_tree, best_score_decision_tree  = perform_grid_search(... ) # TODO

# Do Grid search for Random Forest
grid_random_forest, best_params_random_forest, best_score_random_forest  = perform_grid_search(...) # TODO

# Do Grid search for SVM
grid_svm, best_params_svm, best_score_svm = perform_grid_search(...) # TODO









