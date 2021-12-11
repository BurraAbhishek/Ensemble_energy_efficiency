from operator import pos
import sklearn
import sys
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statistics
from timeit import default_timer as timer
import threading
from threading import Thread

# Uncomment any dataset which you want to use
# The last uncommented dataset will be used.

#dataset_location = "../datasets/iris.csv"
#dataset_location = "../datasets/iris10.csv"
dataset_location = "../datasets/iris100.csv"


# Thread class with return values
# https://stackoverflow.com/a/6894023
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


global classifiers
global results
global performances

classifiers = []
results = []
performances = []


n_algorithms = 5
for i in range(n_algorithms + 1):
    classifiers.append("")
    results.append([])
    performances.append(dict())


def record_performances(classifiers, results, performances, classifier, prediction, duration, position):
    classifiers[position] = (classifier)
    performances[position] = ({
        "model": classifier,
        "accuracy": metrics.accuracy_score(Y_test, prediction),
        "precision": metrics.precision_score(Y_test, prediction, average='weighted'),
        "recall": metrics.recall_score(Y_test, prediction, average='weighted'),
        "F1-score": metrics.f1_score(Y_test, prediction, average='weighted'),
        "duration": duration
    })
    results[position] = list(prediction)
    return classifiers, results, performances

overall_start = timer()


def logisticregression(classifiers, results, performances, X_train, X_test, Y_train, Y_test, position):
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    start = timer()
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, Y_train)
    y_pred_lr = lr.predict(X_test)
    end = timer()
    t = end - start
    record_performances(classifiers, results, performances, "Logistic Regression", y_pred_lr, t, position)


def perceptron(classifiers, results, performances, X_train, X_test, Y_train, Y_test, position):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Perceptron
    start = timer()
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    ppn = Perceptron(eta0=0.1, random_state=1)
    ppn.fit(X_train_std, Y_train)
    y_pred_ppn = ppn.predict(X_test_std)
    end = timer()
    t = end - start
    record_performances(classifiers, results, performances, "Perceptron", y_pred_ppn, t, position)


def svmlinear(classifiers, results, performances, X_train, X_test, Y_train, Y_test, position):
    from sklearn import svm
    start = timer()
    svm_clf = svm.SVC(kernel='linear')
    svm_clf.fit(X_train, Y_train)
    y_pred_svm = svm_clf.predict(X_test)
    end = timer()
    t = end - start
    record_performances(classifiers, results, performances, "Support Vector Machines", y_pred_svm, t, position)


def decisiontree(classifiers, results, performances, X_train, X_test, Y_train, Y_test, position):
    from sklearn.tree import DecisionTreeClassifier
    start = timer()
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,Y_train)
    y_pred_dtree = clf.predict(X_test)
    end = timer()
    t = end - start
    record_performances(classifiers, results, performances, "Decision Tree", y_pred_dtree, t, position)


def randomforest(classifiers, results, performances, X_train, X_test, Y_train, Y_test, position):
    from sklearn.ensemble import RandomForestClassifier
    start = timer()
    rfc = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
    rfc = rfc.fit(X_train, Y_train)
    y_pred_rfc = rfc.predict(X_test)
    end = timer()
    t = end - start
    record_performances(classifiers, results, performances, "Random Forest Classifier", y_pred_rfc, t, position)


def ensemble_vote(p, results):
    a = []
    for j in results:
        a.append(j[p])
    r = statistics.mode(a)
    return r 


if __name__ == "__main__":
    iris_data = pd.read_csv(dataset_location)
    X = iris_data.iloc[:, [0, 1, 2, 3]]
    Y = iris_data.iloc[:, 4]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=1, stratify=Y)

    X_train_list = X_train.values.tolist()
    X_test_list = X_test.values.tolist()
    Y_train_list = Y_train.values.tolist()
    Y_test_list = Y_test.values.tolist()

    t1 = ThreadWithReturnValue(target=logisticregression, args=(classifiers, results, performances, X_train, X_test, Y_train, Y_test, 0, ))
    t2 = ThreadWithReturnValue(target=perceptron, args=(classifiers, results, performances, X_train, X_test, Y_train, Y_test, 1, ))
    t3 = ThreadWithReturnValue(target=svmlinear, args=(classifiers, results, performances, X_train, X_test, Y_train, Y_test, 2, ))
    t4 = ThreadWithReturnValue(target=decisiontree, args=(classifiers, results, performances, X_train, X_test, Y_train, Y_test, 3, ))
    t5 = ThreadWithReturnValue(target=randomforest, args=(classifiers, results, performances, X_train, X_test, Y_train, Y_test, 4, ))
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    
    y_pred_ensemble = []
    start = timer()
    for i in range(0, len(results[0])):
        a = []
        for j in range(0, len(results) - 1):
            a.append(results[j][i])
        y_pred_ensemble.append(statistics.mode(a))

    end = timer()
    t = end - start
    classifiers, results, performances = record_performances(
        classifiers, results, performances, "Ensemble Learning", y_pred_ensemble, t, 5
        )

    overall_end = timer()
    overall_time = overall_end - overall_start

    #print(performances)
    print(pd.DataFrame(list(performances)))
    print("\nTotal time taken:", overall_time)

