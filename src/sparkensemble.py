import sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statistics
from timeit import default_timer as timer
import pyspark
from pyspark.sql import SparkSession
from pyspark import ml
from pyspark.ml.evaluation import MulticlassClassificationEvaluator as evaluator
from pyspark.mllib.evaluation import MulticlassMetrics


# Uncomment any dataset which you want to use
# The last uncommented dataset will be used.

#dataset_location = "../datasets/iris.csv"
#dataset_location = "../datasets/iris10.csv"
dataset_location = "../datasets/iris100.csv"


spark = SparkSession.builder.appName('sparkensemble').getOrCreate()

iris_data = spark.read.csv(dataset_location, header=True)

vectorassembled = ml.feature.VectorAssembler(
    inputCols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    outputCol='X'
    )
iris_data = iris_data.withColumn("sepal_length", iris_data.sepal_length.cast('double'))
iris_data = iris_data.withColumn("sepal_width", iris_data.sepal_width.cast('double'))
iris_data = iris_data.withColumn("petal_length", iris_data.petal_length.cast('double'))
iris_data = iris_data.withColumn("petal_width", iris_data.petal_width.cast('double'))
iris_data = vectorassembled.transform(iris_data)

targets = ml.feature.StringIndexer(inputCol='species', outputCol='Y')
labelConverter = ml.feature.IndexToString(inputCol='Y', outputCol='species')
iris_data = targets.fit(iris_data).transform(iris_data)

#X = iris_data.collect(0, 3)
#Y = iris_data.select('species').collect()

train_df, test_df = iris_data.randomSplit([.50, .50], seed=0)


classifiers = []
results = []
performances = []


def record_performances(classifier, model, duration):
    classifiers.append(classifier)
    e = evaluator(labelCol="Y", predictionCol="prediction")
    performances.append({
        "model": classifier,
        "accuracy": e.evaluate(model, {e.metricName: "accuracy"}),
        "precision": e.evaluate(model, {e.metricName: "weightedPrecision"}),
        "recall": e.evaluate(model, {e.metricName: "weightedRecall"}),
        "F1-score": e.evaluate(model, {e.metricName: "f1"}),
        "duration": duration
    })
    return performances

overall_start = timer()

"""
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
train_df_lr, test_df_lr = train_df, test_df
start = timer()
lr = LogisticRegression(featuresCol="X", labelCol="Y", maxIter=1000)
pipelr = Pipeline(stages=[lr])
pipelr.fit(train_df_lr)
y_pred_lr = pipelr.transform(test_df_lr)
end = timer()
t = end - start
record_performances("Logistic Regression", y_pred_lr, t)

from pyspark.ml.classification import MultilayerPerceptronClassifier as MLP
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

from pyspark.ml.classification import LinearSVC
train_df_svm, test_df_svm = train_df, test_df
start = timer()
svm_clf = LinearSVC(maxIter=1000, regParam=0.1)
svm_clf.fit(train_df_svm)
y_pred_svm = svm_clf.predict(test_df_svm)
end = timer()
t = end - start
record_performances("Support Vector Machines", y_pred_svm, t)
"""

from pyspark.ml.classification import DecisionTreeClassifier
train_df_clf, test_df_clf = train_df, test_df
start = timer()
clf = DecisionTreeClassifier(featuresCol="X", labelCol="Y")
clf = clf.fit(train_df_clf)
y_pred_dtree = clf.transform(test_df_clf)
end = timer()
t = end - start
record_performances("Decision Tree", y_pred_dtree, t)

from pyspark.ml.classification import RandomForestClassifier
train_df_rfc, test_df_rfc = train_df, test_df
start = timer()
rfc = RandomForestClassifier(featuresCol="X", labelCol="Y")
rfc = rfc.fit(train_df_rfc)
y_pred_rfc = rfc.transform(test_df_rfc)
end = timer()
t = end - start
y_pred_rfc.show()
record_performances("Random Forest Classifier", y_pred_rfc, t)

"""
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
"""

overall_end = timer()
overall_time = overall_end - overall_start

print(pd.DataFrame(performances))
print("\nTotal time taken:", overall_time)

