    
from sklearn.preprocessing import OneHotEncoder
import pandas
import gzip
import shutil

def fetch_data_preprocessed(type):
    if type == 'TEST':
        ###########################################################
        # Extract Data From Files Into Lables and Attribute Values
        ###########################################################
        with gzip.open('../Resources/UNSW_NB15_testing-set.csv.gz', 'rb') as f_in:
            with open('testing-data.csv', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        testing_data = pandas.read_csv('./testing-data.csv').values
        X_test = testing_data[:,1:-2]
        Y_test = testing_data[:,-2:-1]


        ###########################################################
        # Preprocess Data - Convert Categorical Data to Numerical
        ###########################################################

        # Get the portion of the data which is categorical
        X_test_categorical_piece = X_test[:,1:4]

        # One-Hot Encode the categorical portions of the data
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(X_test_categorical_piece)
        X_test_categorical_piece_encoded = encoder.transform(X_test_categorical_piece).toarray()

        # Replace categorical parts with one-hot encoding
        X_test_encoded = []
        for i in range(len(X_test)):
            row = [X_test[i][0]]
            row.extend(X_test_categorical_piece_encoded[i][:].tolist())
            row.extend(X_test[i][4:].tolist())
            X_test_encoded.append(row)

        return X_test_encoded, Y_test

    elif type == 'TRAIN':
        ###########################################################
        # Extract Data From Files Into Lables and Attribute Values
        ###########################################################
        with gzip.open('../Resources/UNSW_NB15_training-set.csv.gz', 'rb') as f_in:
            with open('training-data.csv', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        training_data = pandas.read_csv('./training-data.csv').values
        X_train = training_data[:,1:-2]
        Y_train = training_data[:,-2:-1]

        ###########################################################
        # Preprocess Data - Convert Categorical Data to Numerical
        ###########################################################

        # Get the portion of the data which is categorical
        X_train_categorical_piece = X_train[:,1:4]


        # One-Hot Encode the categorical portions of the data
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(X_train_categorical_piece)
        X_train_categorical_piece_encoded = encoder.transform(X_train_categorical_piece).toarray()

        # Replace categorical parts with one-hot encoding
        X_train_encoded = []
        for i in range(len(X_train)):
            row = [X_train[i][0]]
            row.extend(X_train_categorical_piece_encoded[i][:].tolist())
            row.extend(X_train[i][4:].tolist())
            X_train_encoded.append(row)

        return X_train_encoded, Y_train

    else:
        raise ValueError('Invalid type! Must be "TEST" or "TRAIN"')

