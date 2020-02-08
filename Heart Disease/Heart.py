# we will make an app that will guess how old someone is given information about their heart

# lets import some stuff we need
import pandas # reads in the csv file for us
import numpy as np # allows us to use arrays in python
import sklearn # allows us to make our model
from sklearn.neighbors import KNeighborsClassifier
import pickle # allows us to save our model
from time import sleep
from datetime import datetime

'''this will let us put a timestamp on our logs
Since we will let this train for a few million loops, i want to be able to print an update every x ammount of loops
and i want a time stand on those updates '''


data = pandas.read_csv('heart.csv') # reads in the csv file
labelToBePredicted = "target"

x = np.array(data.drop([labelToBePredicted], 1)) # an array of our attributes except what we want to find (labels)
y = np.array(data[labelToBePredicted]) # an array of only the label we want to find

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)  # test size means 90% of the data will be used to train and 10% will be used to test


def TimeStamp(neighborIndex, loopIndex):
    currentTime = datetime.now()
    print("\nDate: " + str(currentTime.month) + "/" + str(currentTime.day) + "/" + str(currentTime.year))
    print("Time: " + str(currentTime.hour) + ":" + str(currentTime.minute) + ":" + str(currentTime.second))
    print("On Neighbor " + str(neighborIndex) + " on loop " + str(loopIndex) + "\n")

maxNumOfNeighbors = 22
numOfLoopsPerNeighbor = 5000000
bestAccuracyThusFar = 0
neighborThatBestAccuracyWasOn = 0

for i in range(maxNumOfNeighbors):
    if i % 2 == 1:
        for j in range(numOfLoopsPerNeighbor):
            model = KNeighborsClassifier(n_neighbors=i)
            model.fit(x_train, y_train)
            currentAccuracy = model.score(x_test, y_test)
            if j % 100000 == 0:
                TimeStamp(i, j)
            if currentAccuracy > bestAccuracyThusFar:
                with open("Heart.pickle", "wb") as file:
                    pickle.dump(model, file)

                bestAccuracyThusFar = currentAccuracy
                neighborThatBestAccuracyWasOn = i
                print("\n\n ~ ~ BEST SO FAR ~ ~ ", "\nAccuracy: " + str(bestAccuracyThusFar))
                print( "Neighbor: " + str(neighborThatBestAccuracyWasOn), "\n ~ ~ ~ ~ ~ ~ ~ ~ ~ ~\n\n")
            sleep(0.01)
print("\n\n === TRAIN RESULTS ===")
print("Top Accuracy: " + str(bestAccuracyThusFar) + " with " + str(neighborThatBestAccuracyWasOn) + " Neighbors")


pickle_in = open("Heart.pickle", "rb")
model = pickle.load(pickle_in)

predicted = model.predict(x_test)

names = ["has disease", "no disease"] # helps me tell if they have the disease or not. other wise i would just see 0 and 1
print("\n\nTests:")
for i in range(len(x_test)):
    print("\nData: ", x_test[i], "\nComputer's Prediction: ", names[predicted[i]], "\nActual Value: ", names[y_test[i]])


