from sklearn.model_selection import train_test_split
from q1_question import data_preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


def data_splits(X, y):
    """
    Split the 'features' and 'labels' data into training and testing sets.
    Input(s): X: features (pd.DataFrame), y: labels (pd.DataFrame)
    Output(s): X_train, X_test, y_train, y_test
    """
    # Use random_state = 0 in the train_test_split
    # TODO write data split here

    return X_train, X_test, y_train, y_test


def normalize_features(X_train, X_test):
    """
    Take the input data and normalize the features.
    Input: X_train: features for train,  X_test: features for test (pd.DataFrame)
    Output: X_train_scaled, X_test_scaled (pd.DataFrame) the same shape of X_train and X_test
    """
    # TODO write normalization here

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
        cls = ... # TODO call classifier here
    elif model_name == 'Random Forest':
        cls = ... # TODO call classifier here
    elif model_name == 'SVM':
        cls = ... # TODO call classifier here

    ... # TODO trian the model

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
        y_train_pred = ... # TODO predict y
        y_test_pred = . .. # TODO predict y

        # Calculate accuracy
        train_accuracy = ... # TODO find accuracy
        test_accuracy = ... # TODO find accuracy

        # Calculate F1-score
        train_f1 = ... # TODO find f1_score
        test_f1 = ... # TODO find f1_score

        # Store predictions
        y_train_pred_dict[model_name] = ... # TODO
        y_test_pred_dict[model_name] = ... # TODO

        # Store the evaluation metrics
        evaluation_results[model_name] = {
            'Train Accuracy': ... # TODO ,
            'Test Accuracy': ...  # TODO ,
            'Train F1 Score': ... # TODO ,
            'Test F1 Score': ...  # TODO
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
        y_train_pred = ... # TODO compelete it
        y_test_pred = ... # TODO compelete it

        # Print classification report for training set
        print("\nTraining Set Classification Report:")
        # TODO write Classification Report train

        # Print confusion matrix for training set
        print("Training Set Confusion Matrix:")
        # TODO write Confusion Matrix train

        # Print classification report for testing set
        print("\nTesting Set Classification Report:")
        # TODO write Classification Report test

        # Print confusion matrix for testing set
        print("Testing Set Confusion Matrix:")
        # TODO write Confusion Matrix test



X, y = ... # TODO call data preprocessing from q1
X_train, X_test, y_train, y_test = ... # TODO split data
X_train_scaled, X_test_scaled = ... # TODO normalize data

cls_decision_tree = ... # TODO train the model
cls_randomforest = ... # TODO train the model
cls_svm = ... # TODO train the model

# Define a dictionary of model name and their trained model
trained_models = {
        'Decision Tree': cls_decision_tree,
        'Random Forest': cls_randomforest,
        'SVM': cls_svm }

# predict labels and calculate accuracy and F1score
y_train_pred_dict, y_test_pred_dict, evaluation_results = eval_model(trained_models, X_train_scaled, X_test_scaled, y_train, y_test)

# classification report and calculate confusion matrix
report_model(y_train, y_test, y_train_pred_dict, y_test_pred_dict)



















