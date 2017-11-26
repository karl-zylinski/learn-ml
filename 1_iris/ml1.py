from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import pprint

iris = datasets.load_iris()
learn_data = iris.data[0:148]
learn_targets = iris.target[0:148]
predict_data = iris.data[148:150]
predict_targets = iris.target[148:150]
clf = LogisticRegression()
print(learn_targets)
clf.fit(learn_data, learn_targets)
X_new = [[ 5.0,  3.6,  1.3,  0.25]]
result = clf.predict(X_new)
print(clf.predict_proba(X_new))
print(result)
print(predict_targets)