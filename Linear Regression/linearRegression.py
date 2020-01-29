import pandas as pd #allows us to read in the data sets
import numpy as np # allows us to use arrays in python
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle
from time import sleep
from datetime import datetime

def TimeStamp():
    currentTime = datetime.now()
    print("Date: " + str(currentTime.month) + "/" + str(currentTime.day) + "/" + str(currentTime.year))
    print("Time: " + str(currentTime.hour) + ":" + str(currentTime.minute) + ":" + str(currentTime.second) + "\n")

data = pd.read_csv("student-mat.csv", sep=";") #gets the data from the file and seperates the data using semicolons... usually its commas so you have to tell it to use ';'

data = data[["G1", "G2", 'studytime', "failures", "absences", "health", "traveltime", 'G3']] #allows us to only use SOME of the atributes, not all of them


predict = "G3" #what you want to look for, called a label

x = np.array(data.drop([predict], 1)) # an array of our attributes except what we want to find (labels)
y = np.array(data[predict]) # an array of only the label we want to find

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)  # test size means 90% of the data will be used to train and 10% will be used to test


bestScoreSoFar = 0
bestTestSize = 0
loopLength = 2000000
for j in range(29):
    if j == 0:
        j = 10
    j = j/100
    print(j)
    for i in range(loopLength):
        sleep(0.075)
        # we are going to split these up into 4 different arrays
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=j) # test size means 90% of the data will be used to train and 10% will be used to test

        linear = linear_model.LinearRegression() # allows us to use linear regression

        linear.fit(x_train, y_train) # finds the best fit line
        accuracyOFModel = linear.score(x_test, y_test)

        if i % 500 == 0:
            print( "\n\nCurrent Accuracy: ",  accuracyOFModel, f'{i:,}', "/", f'{loopLength:,}', " \nwith a test size of ", j )
            TimeStamp()

        if accuracyOFModel > bestScoreSoFar:
            bestTestSize = j
            bestScoreSoFar = accuracyOFModel
            print("\nbest is now: ", bestScoreSoFar, " with a test size of ", bestTestSize)
            with open("studentModel.pickle", "wb") as file:
                pickle.dump(linear, file) 

print("RESULTS... BEST ACCURACY: ", bestScoreSoFar, " with a test size of ", bestTestSize)

pickle_in = open("studentModel.pickle", "rb")
linear = pickle.load(pickle_in)

predictions = linear.predict(x_test)


for x in range(len(predictions)):
    print("\nData: " , x_test[x] , "\nPrediction: " , predictions[x], "\nActual Answer: " , y_test[x])

attribute = 'G1'
style.use("ggplot") #makes our grid look nice
pyplot.scatter(data[attribute], data[predict]) #gives it x and y values
pyplot.xlabel(attribute)
pyplot.ylabel("Final Grade")

pyplot.show()