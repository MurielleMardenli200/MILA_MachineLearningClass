from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from q1 import data_preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def data_splits(X, y):
    """
    Split the 'features' and 'labels' data into training and testing sets.
    Input(s): X: features (pd.DataFrame), y: labels (pd.DataFrame)
    Output(s): X_train, X_test, y_train, y_test
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y.squeeze(), test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test


def normalize_features(X_train, X_test):
    """
    Take the input data and normalize the features.
    Input: X_train: features for train,  X_test: features for test (pd.DataFrame)
    Output: X_train_scaled, X_test_scaled (pd.DataFrame) the same shape of X_train and X_test
    """

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def train_model(model_name, X_train_scaled, y_train):
    '''
    inputs:
       - model_name: the name of learning algorithm to be trained
       - X_train: features training set
       - y_train: label training set
    output: cls: the trained model
    '''


    if model_name == 'Decision Tree':
        cls = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=10, min_samples_leaf=4, min_samples_split= 2)
    elif model_name == 'Random Forest':
        cls = RandomForestClassifier(random_state=0, bootstrap=True, max_depth=20, n_estimators=50, min_samples_leaf=4)
    elif model_name == 'SVM':
        cls = SVC(random_state=0, C=4, kernel='rbf', shrinking=True)

    cls.fit(X_train_scaled, y_train)

    return cls


def eval_model(trained_models, X_train, X_test, y_train, y_test):
    '''
    inputs:
       - trained_models: a dictionary of the trained models,
       - X_train: features training set
       - X_test: features test set
       - y_train: label training set
       - y_test: label test set
    outputs:
        - y_train_pred_dict: a dictionary of label predicted for train set of each model
        - y_test_pred_dict: a dictionary of label predicted for test set of each model
        - a dict of accuracy and f1_score of train and test sets for each model
    '''
    evaluation_results = {}
    y_train_pred_dict = {
        'Decision Tree': None,
        'Random Forest': None,
        'SVM': None}
    y_test_pred_dict = {
        'Decision Tree': None,
        'Random Forest': None,
        'SVM': None}

    # Loop through each trained model
    for model_name, model in trained_models.items():
        # Predictions for training and testing sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate accuracy
        train_accuracy = accuracy_score(y_true=y_train, y_pred=y_train_pred)
        test_accuracy = accuracy_score(y_true=y_test, y_pred=y_test_pred)

        # Calculate F1-score
        train_f1 = f1_score(y_true=y_train, y_pred=y_train_pred)
        test_f1 =  f1_score(y_true=y_test, y_pred=y_test_pred)

        # Store predictions
        y_train_pred_dict[model_name] = y_train_pred
        y_test_pred_dict[model_name] = y_test_pred

        # Store the evaluation metrics
        evaluation_results[model_name] = {
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy,
            'Train F1 Score': train_f1 ,
            'Test F1 Score': test_f1
        }

    # Return the evaluation results
    return y_train_pred_dict, y_test_pred_dict, evaluation_results


def report_model(y_train, y_test, y_train_pred_dict, y_test_pred_dict):
    '''
    inputs:
        - y_train: label training set
        - y_test: label test set
        - y_train_pred_dict: a dictionary of label predicted for train set of each model, len(y_train_pred_dict.keys)=3
        - y_test_pred_dict: a dictionary of label predicted for test set of each model, len(y_train_pred_dict.keys)=3
    '''

    # Loop through each trained model
    for model_name, model in trained_models.items():
        print(f"\nModel: {model_name}")

        # Predictions for training and testing sets
        y_train_pred = y_train_pred_dict[model_name]
        y_test_pred = y_test_pred_dict[model_name]

        # Print classification report for training set
        print("\nTraining Set Classification Report:")
        # TODO write Classification Report train
        print(classification_report(y_true=y_train, y_pred=y_train_pred))

        # Print confusion matrix for training set
        print("Training Set Confusion Matrix:")
        # TODO write Confusion Matrix train
        print(confusion_matrix(y_true=y_train, y_pred=y_train_pred))

        # Print classification report for testing set
        print("\nTesting Set Classification Report:")
        # TODO write Classification Report test
        print(classification_report(y_true=y_test, y_pred=y_test_pred))

        # Print confusion matrix for testing set
        print("Testing Set Confusion Matrix:")
        # TODO write Confusion Matrix test
        print(confusion_matrix(y_true=y_test, y_pred=y_test_pred))



X, y = data_preprocessing()
X_train, X_test, y_train, y_test = data_splits(X, y)
X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

# label numerical
cls_decision_tree = train_model('Decision Tree', X_train_scaled, y_train)
cls_randomforest = train_model('Random Forest', X_train_scaled, y_train)
cls_svm = train_model('SVM', X_train_scaled, y_train)

# Define a dictionary of model name and their trained model
trained_models = {
        'Decision Tree': cls_decision_tree,
        'Random Forest': cls_randomforest,
        'SVM': cls_svm }

# predict labels and calculate accuracy and F1score
# y_train_pred_dict, y_test_pred_dict, evaluation_results = eval_model(trained_models, X_train_scaled, X_test_scaled, y_train, y_test)

# print('\nevaluation results')
# print(evaluation_results)

print('\nclassification report')
# classification report and calculate confusion matrix
# report_model(y_train, y_test, y_train_pred_dict, y_test_pred_dict)


model_names = list(trained_models.keys())
accuracies = [0.89229, 0.89934, 0.892634]

# # Plotting
# plt.figure(figsize=(8, 6))
# plt.bar(model_names, accuracies, color='skyblue', edgecolor='black')
# plt.title('Test Accuracy for Different Models')
# plt.xlabel('Model Names')
# plt.ylabel('Accuracy')

# # Adjust y-axis range to focus on differences
# plt.ylim([0.87, 0.91])

# # Add grid lines for clarity
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()



def evaluate_hyperparameters(X_train_scaled, y_train):
    results = {}

    # Decision Tree: max_depth = {None, 5, 10}
    decision_tree_accuracies = []
    for depth in [None, 5, 10]:
        model = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=depth)
        model.fit(X_train_scaled, y_train)
        train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
        decision_tree_accuracies.append(train_accuracy)
    results['Decision Tree'] = decision_tree_accuracies
    print('\n decision tree accuracy')
    print(decision_tree_accuracies)

    # Random Forest: n_estimators = {5, 10, 30}
    random_forest_accuracies = []
    for n_estimators in [5, 10, 30]:
        model = RandomForestClassifier(random_state=0, n_estimators=n_estimators)
        model.fit(X_train_scaled, y_train)
        train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
        random_forest_accuracies.append(train_accuracy)
    results['Random Forest'] = random_forest_accuracies
    print('\n forest tree accuracy')
    print(random_forest_accuracies)

    # SVM: kernel = {'linear', 'poly', 'rbf'}
    svm_accuracies = []
    for kernel in ['linear', 'poly', 'rbf']:
        model = SVC(random_state=0, kernel=kernel)
        model.fit(X_train_scaled, y_train)
        train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
        svm_accuracies.append(train_accuracy)
    results['SVM'] = svm_accuracies
    print('\n svm accuracy')
    print(svm_accuracies)

    return results


def plot_hyperparameter_results(results):
    # Plot results for each model
    for model_name, accuracies in results.items():
        plt.figure(figsize=(8, 6))
        if model_name == 'Decision Tree':
            x_values = ['None', '5', '10']
            x_label = 'Max Depth'
        elif model_name == 'Random Forest':
            x_values = ['5', '10', '30']
            x_label = 'Number of Estimators'
        elif model_name == 'SVM':
            x_values = ['Linear', 'Poly', 'RBF']
            x_label = 'Kernel Type'

        plt.plot(x_values, accuracies, color='skyblue')
        plt.title(f'{model_name} Hyperparameter Tuning')
        plt.xlabel(x_label)
        plt.ylabel('Training Accuracy')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()


# Main Script
X, y = data_preprocessing()
X_train, X_test, y_train, y_test = data_splits(X, y)
X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

# Evaluate models with varying hyperparameters
hyperparameter_results = evaluate_hyperparameters(X_train_scaled, y_train)

# Plot results
plot_hyperparameter_results(hyperparameter_results)