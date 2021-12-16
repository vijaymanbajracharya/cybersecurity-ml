import decision_tree
import random_forest
import linear_svm
import knn
import naive_bayes
from cross_validation import cross_validate

if __name__ == '__main__':
    print("Some of these algorithms can take a long time to run since the dataset is large\r\n")
    # Perform Cross Validation to find best depth
    possible_depths = [10, 20]
    best_param = cross_validate(possible_depths, "DT")

    DT = decision_tree.run(max_depth=best_param)
    print(DT)

    # Perform Cross Validation to find best number of estimators
    # possible_estimators = [50, 100]
    # best_param = cross_validate(possible_estimators, "RF")

    RF = random_forest.run(n_estimators=100)
    print(RF)

    SVM = linear_svm.run()
    print(SVM)

    # Perform Cross Validation to find best number of neighbors
    # possible_nn = [3, 5]
    # best_param = cross_validate(possible_nn, "KNN")

    KNN = knn.run(n_neighbors=5)
    print(KNN)

    NB = naive_bayes.run()
    print(NB)
