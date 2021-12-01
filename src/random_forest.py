from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from classifier_analysis import ClassifierAnalysis
import numpy as np
import pandas
import gzip
import shutil

def run():
    ###########################################################
    # Extract Data From Files Into Lables and Attribute Values
    ###########################################################
    with gzip.open('../Resources/UNSW_NB15_training-set.csv.gz', 'rb') as f_in:
        with open('training-data.csv', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    with gzip.open('../Resources/UNSW_NB15_testing-set.csv.gz', 'rb') as f_in:
        with open('testing-data.csv', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    training_data = pandas.read_csv('./training-data.csv').values
    X_train = training_data[:,1:-2]
    Y_train = training_data[:,-2:-1]

    testing_data = pandas.read_csv('./testing-data.csv').values
    X_test = testing_data[:,1:-2]
    Y_test = testing_data[:,-2:-1]


    ###########################################################
    # Preprocess Data - Convert Categorical Data to Numerical
    ###########################################################

    # Get the portion of the data which is categorical
    X_train_categorical_piece = X_train[:,1:4]
    X_test_categorical_piece = X_test[:,1:4]

    # One-Hot Encode the categorical portions of the data
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(X_train_categorical_piece)
    X_train_categorical_piece_encoded = encoder.transform(X_train_categorical_piece).toarray()
    X_test_categorical_piece_encoded = encoder.transform(X_test_categorical_piece).toarray()

    # Replace categorical parts with one-hot encoding
    X_train_encoded = []
    for i in range(len(X_train)):
        row = [X_train[i][0]]
        row.extend(X_train_categorical_piece_encoded[i][:].tolist())
        row.extend(X_train[i][4:].tolist())
        X_train_encoded.append(row)

    X_test_encoded = []
    for i in range(len(X_test)):
        row = [X_test[i][0]]
        row.extend(X_test_categorical_piece_encoded[i][:].tolist())
        row.extend(X_test[i][4:].tolist())
        X_test_encoded.append(row)

    ########################################
    # Create Classifier
    ########################################
    classifier = RandomForestClassifier(n_estimators=5, max_depth=20)
    classifier = classifier.fit(X_train_encoded, Y_train.flatten())

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

    return ClassifierAnalysis('Random Forest', test_error, train_error, classification_error_by_label)
