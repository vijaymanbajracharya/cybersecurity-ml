from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from preprocess_data import fetch_data_preprocessed

#########################################################
# Use KFold Cross Validation to find best parameter
#########################################################

def cross_validate(params:list, classifier="DT", n_splits=5):
    X_train_encoded, Y_train, _, _ = fetch_data_preprocessed()

    kf = KFold(n_splits=n_splits)
    least_val_error = 1.0
    hyperparam_index = None

    for index, param in enumerate(params):
        val_error = 0.0
        for train_index, val_index in kf.split(X_train_encoded):
            X_train_split, X_val_split = X_train_encoded[train_index], X_train_encoded[val_index]
            y_train_split, y_val_split = Y_train[train_index], Y_train[val_index]

            if classifier == "DT":
                classifier = DecisionTreeClassifier(max_depth=param, min_samples_leaf=8, min_samples_split=40)
                classifier = classifier.fit(X_train_split, y_train_split)
            elif classifier == "KNN":
                classifier = KNeighborsClassifier(n_neighbors=param, weights='distance', algorithm='kd_tree', leaf_size=15, n_jobs=-1)
                classifier = classifier.fit(X_train_split, y_train_split.flatten())
            elif classifier == "RF":
                classifier = BalancedRandomForestClassifier(n_estimators=param, max_depth=3)
                classifier = classifier.fit(X_train_split, y_train_split.flatten())
            else:
                break

            val_pred = classifier.predict(X_val_split)
            val_error += 1 - accuracy_score(y_val_split, val_pred)
        
        val_error = val_error / kf.get_n_splits(X_train_encoded)
        if val_error <= least_val_error:
            least_val_error = val_error
            hyperparam_index = index

    best_param = params[hyperparam_index]
    return best_param
