    
from scipy.sparse.construct import rand
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import VarianceThreshold
import pandas
import gzip
import shutil
import numpy as np

def fetch_data_preprocessed():
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

    X_train_encoded = np.array(X_train_encoded)

    X_test_encoded = []
    for i in range(len(X_test)):
        row = [X_test[i][0]]
        row.extend(X_test_categorical_piece_encoded[i][:].tolist())
        row.extend(X_test[i][4:].tolist())
        X_test_encoded.append(row)

    X_test_encoded = np.array(X_test_encoded)
    
    ###########################################################
    # Under/Over Sample Data By Label
    ###########################################################

    under_sample_dict = {
        'Normal' : 10000,
        'Backdoor' : 583,
        'Analysis' : 677,
        'Fuzzers' : 6062,
        'Shellcode' : 378,
        'Reconnaissance' : 3496,
        'Exploits' : 10000,
        'DoS' : 4089,
        'Worms' : 44,
        'Generic' : 10000
    }
    
    over_sample_dict = {
        'Normal' : 10000,
        'Backdoor' : 10000,
        'Analysis' : 10000,
        'Fuzzers' : 10000,
        'Shellcode' : 10000,
        'Reconnaissance' : 10000,
        'Exploits' : 10000,
        'DoS' : 10000,
        'Worms' : 10000,
        'Generic' : 10000
    }

    # sampler = RandomUnderSampler(random_state=0, sampling_strategy=under_sample_dict)
    # X_train_encoded, Y_train = sampler.fit_resample(X_train_encoded, Y_train)

    # sampler = RandomOverSampler(random_state=0, sampling_strategy=over_sample_dict)
    # X_train_encoded, Y_train = sampler.fit_resample(X_train_encoded, Y_train)

    return X_train_encoded, Y_train, X_test_encoded, Y_test

