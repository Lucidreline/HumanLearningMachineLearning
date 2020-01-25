import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import pickle

data = pd.read_csv("car.data")
print(data.head())


convertToNumbers = preprocessing.LabelEncoder()

buying= convertToNumbers.fit_transform(list(data["buying"])) # creates a list where all the non numerical values turn into values
maint= convertToNumbers.fit_transform(list(data["maint"]))
door= convertToNumbers.fit_transform(list(data["door"]))
persons= convertToNumbers.fit_transform(list(data["persons"]))
lug_boot= convertToNumbers.fit_transform(list(data["lug_boot"]))
safety= convertToNumbers.fit_transform(list(data["safety"]))
cls = convertToNumbers.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot))
y = list(cls)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)  # test size means 90% of the data will be used to train and 10% will be used to test

loopSize = 30000
numberOfNeighborsToTry = 11
bestAccuracyThusFar = 0
bestNumberOfNeighbors = 0
'''
for i in range(numberOfNeighborsToTry):
    i = i + 1
    if i % 2 == 1:
        for j in range(loopSize):
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)  # test size means 90% of the data will be used to train and 10% will be used to test

            model = KNeighborsClassifier(n_neighbors=i)

            model.fit(x_train, y_train)
            currentAccuracy = model.score(x_test, y_test)
            print("Trying " + str(i) + " neighbors... Accuracy: " + str(currentAccuracy), f'{j:,}', "/", f'{loopSize:,}')

            if currentAccuracy > bestAccuracyThusFar:
                bestAccuracyThusFar = currentAccuracy
                bestNumberOfNeighbors = i
                print("\n\nBEST IS NOW: " + str(bestAccuracyThusFar) + "\n\n")
                with open("cars.pickle", "wb") as file:
                    pickle.dump(model, file)
'''

pickle_in = open("cars.pickle", "rb")
model = pickle.load(pickle_in)


print("Best Accuracy is: ", bestAccuracyThusFar, " with ", bestNumberOfNeighbors, " neighbors")
print("Confirm Accuracy: ", model.score(x_test, y_test))




predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for i in range(len(x_test)):
    print("\nData: ", x_test[i], "\nPredicted: ", names[predicted[i]], "\nActual: ", names[y_test[i]])