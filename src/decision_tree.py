from sklearn.tree import DecisionTreeClassifier
from classifier_analysis import ClassifierAnalysis
from preprocess_data import fetch_data_preprocessed
import numpy as np

def run():
    ########################################
    # Fetch and Preprocess Data
    ########################################
    X_train_encoded, Y_train = fetch_data_preprocessed(type='TRAIN')
    X_test_encoded, Y_test = fetch_data_preprocessed(type='TEST')

    print(len(X_train_encoded[0]), len(X_test_encoded[0]))
    ########################################
    # Create Classifier
    ########################################
    classifier = DecisionTreeClassifier(max_depth=20)
    classifier = classifier.fit(X_train_encoded, Y_train)

    ########################################
    # Generate Classification Results
    ########################################
    # Calculate Testing Error and Make Classification Error By Label Dictionary
    test_predictions = []
    for row in X_test_encoded:
        test_predictions.append(classifier.predict([row])[0])

    classification_error_by_label = {}
    incorrect_count_test = 0
    for i in range(len(test_predictions)):
        if Y_test[i][0] not in classification_error_by_label: # Add label to dict
            classification_error_by_label[Y_test[i][0]] = [0, 0]

        if test_predictions[i] != Y_test[i]: # Increment incorrect_label count, Increment incorrect count by label
            incorrect_count_test += 1
            classification_error_by_label[Y_test[i][0]][1] += 1
        else: # Increment correct count by label
            classification_error_by_label[Y_test[i][0]][0] += 1


    test_error = incorrect_count_test / len(Y_test)

    # Calculate Training Error
    train_predictions = []
    for row in X_train_encoded:
        train_predictions.append(classifier.predict([row])[0])

    incorrect_count_train = 0
    for i in range(len(train_predictions)):
        if train_predictions[i] != Y_train[i]: incorrect_count_train += 1

    train_error = incorrect_count_train / len(Y_train)

    return ClassifierAnalysis('Decision Tree', test_error, train_error, classification_error_by_label)
