import sklearn import tree
import numpy as np
import gzip
import shutil

###########################################################
# Extract Data From Files Into Lables and Attribute Values
###########################################################
with gzip.open('../Resources/UNSW_NB15_training-set.csv.gz', 'rb') as f_in:
    with open('training-data.csv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

with gzip.open('../Resources/UNSW_NB15_testing-set.csv.gz', 'rb') as f_in:
    with open('testing-data.csv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

training_data = np.genfromtxt('src/training-data.csv', delimeter=',')
X_train = training_data[:-2]
Y_train = training_data[-2:-1]

testing_data = np.genfromtxt('src/testing-data.csv', delimeter=',')
X_test = training_data[:-2]
Y_test = training_data[-2:-1]



###########################################################
# Preprocess Data - Convert Numerical Data to Categorical
###########################################################



########################################
# Create Classifier
########################################
# classifier = tree.DecisionTreeClassifier()



########################################
# Generate Classification Results
########################################