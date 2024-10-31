import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, fbeta_score
from q1 import data_preprocessing
from q2 import data_splits, normalize_features

# Step 1: Create smaller hyperparameter grids for each model
param_grid_decision_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 4],
}

param_grid_random_forest = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'bootstrap': [True]
}

param_grid_svm = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1],
    'gamma': ['scale', 'auto']
}

# Step 2: Initialize classifiers with random_state=0
decision_tree = DecisionTreeClassifier(random_state=0)
random_forest = RandomForestClassifier(random_state=0)
svm = SVC(random_state=0)

# Step 3: Create a scorer using F-beta score with beta=0.5
scorer = make_scorer(fbeta_score, beta=0.5)

# Step 4: Perform grid search for each model using 9-fold StratifiedKFold cross-validation
def perform_grid_search(model, X_train, y_train, params):
    # Define the cross-validation strategy
    strat_kfold = StratifiedKFold(n_splits=9, shuffle=True, random_state=0)

    # Grid search for the model with reduced parallelism and verbosity
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring=scorer,
        cv=strat_kfold,
        n_jobs=1,  # Reduced parallelism to avoid memory issues
        verbose=2  # Increased verbosity to monitor progress
    )
    
    # Fit to the data
    grid_search.fit(X_train, y_train)

    best_param = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best parameters are:", best_param)
    print("Best score is:", best_score)

    # Return the fitted grid search objects
    return grid_search, best_param, best_score

# Load data
X, y = data_preprocessing()
X_train, X_test, y_train, y_test = data_splits(X, y)
X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

# Do Grid search for Decision Tree
grid_decision_tree, best_params_decision_tree, best_score_decision_tree = perform_grid_search(
    decision_tree, X_train_scaled, y_train, param_grid_decision_tree
)

# Do Grid search for Random Forest
grid_random_forest, best_params_random_forest, best_score_random_forest = perform_grid_search(
    random_forest, X_train_scaled, y_train, param_grid_random_forest
)

# Do Grid search for SVM
grid_svm, best_params_svm, best_score_svm = perform_grid_search(
    svm, X_train_scaled, y_train, param_grid_svm
)
