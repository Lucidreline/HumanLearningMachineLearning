import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_breast_cancer() #loads in some data that was in sklearn

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

classes = ["It's bad :(", "You'll Survive"]

model = svm.SVC(kernel="linear", C=2)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
