import sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statistics
from timeit import default_timer as timer


# Uncomment any dataset which you want to use
# The last uncommented dataset will be used.

#dataset_location = "../datasets/iris.csv"
#dataset_location = "../datasets/iris10.csv"
dataset_location = "../datasets/iris100.csv"


iris_data = pd.read_csv(dataset_location)
X = iris_data.iloc[:, [0, 1, 2, 3]]
Y = iris_data.iloc[:, 4]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=1, stratify=Y)

classifiers = []
results = []
performances = []


def record_performances(classifier, prediction, duration):
    classifiers.append(classifier)
    performances.append({
        "model": classifier,
        "accuracy": metrics.accuracy_score(Y_test, prediction),
        "precision": metrics.precision_score(Y_test, prediction, average='weighted'),
        "recall": metrics.recall_score(Y_test, prediction, average='weighted'),
        "F1-score": metrics.f1_score(Y_test, prediction, average='weighted'),
        "duration": duration
    })
    results.append(prediction)

overall_start = timer()

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
start = timer()
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, Y_train)
y_pred_lr = lr.predict(X_test)
end = timer()
t = end - start
record_performances("Logistic Regression", y_pred_lr, t)

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
record_performances("Perceptron", y_pred_ppn, t)

from sklearn import svm
start = timer()
svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(X_train, Y_train)
y_pred_svm = svm_clf.predict(X_test)
end = timer()
t = end - start
record_performances("Support Vector Machines", y_pred_svm, t)

from sklearn.tree import DecisionTreeClassifier
start = timer()
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,Y_train)
y_pred_dtree = clf.predict(X_test)
end = timer()
t = end - start
record_performances("Decision Tree", y_pred_dtree, t)

from sklearn.ensemble import RandomForestClassifier
start = timer()
rfc = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
rfc = rfc.fit(X_train, Y_train)
y_pred_rfc = rfc.predict(X_test)
end = timer()
t = end - start
record_performances("Random Forest Classifier", y_pred_rfc, t)


y_pred_ensemble = []
start = timer()
for i in range(0, len(results[0])):
    a = []
    for j in results:
        a.append(j[i])
    y_pred_ensemble.append(statistics.mode(a))
end = timer()
t = end - start
record_performances("Ensemble Learning", y_pred_ensemble, t)


overall_end = timer()
overall_time = overall_end - overall_start

print(pd.DataFrame(performances))
print("\nTotal time taken:", overall_time)