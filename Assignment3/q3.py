from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from q1 import data_preprocessing
from q2 import data_splits, normalize_features, report_model

param_grid_decision_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 20],
    'min_samples_leaf': [3, 4, 6],
    'min_samples_split': [2, 3]
}

param_grid_random_forest = {
    'n_estimators': [50, 70],
    'max_depth': [10, 20],
    'min_samples_leaf': [3, 4],
    'bootstrap': [True, False],
}

param_grid_svm = {
    'kernel': ['rbf', 'poly'],
    'shrinking': [True, False],
    'C': [3, 4]
}

decision_tree = DecisionTreeClassifier(random_state=0)
random_forest = RandomForestClassifier(random_state=0)
svm = SVC(random_state=0)

scorer = 'accuracy'

def perform_grid_search(model, X_train, y_train, params):
    # Define the cross-validation strategy
    strat_kfold = StratifiedKFold(n_splits=9, shuffle=True, random_state=0)

    # Grid search for the model with reduced parallelism and verbosity
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring=scorer,
        cv=strat_kfold,
        n_jobs=-1,
        verbose=2
    )
    
    # Fit to the data
    grid_search.fit(X_train, y_train)

    best_param = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best parameters are:", best_param)
    print("Best score is:", best_score)

    # Return the fitted grid search objects
    return grid_search, best_param, best_score

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
