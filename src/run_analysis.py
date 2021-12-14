import decision_tree
import random_forest
import linear_svm
import knn
import naive_bayes

if __name__ == '__main__':
    DT = decision_tree.run()
    print(DT)

    RF = random_forest.run()
    print(RF)

    SVM = linear_svm.run()
    print(SVM)

    KNN = knn.run()
    print(KNN)

    NB = naive_bayes.run()
    print(NB)
