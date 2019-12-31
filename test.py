import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style
from sklearn.utils import shuffle
import tensorflow
import keras

# This will give the app access to the csv value
# The values are supposed to be seperated by commas but since they used semi colons, we will also use semicolons
data = pd.read_csv("student-mat.csv", sep=";")

# Prints the first 5
#print(data.head())

# This will only look at these values
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#print(data.head())

# I am going to want to predict this value
predict = "G3"

# x will be an array of all of the values expet the predict values so that the computer wont see the values ahead of time
x = np.array(data.drop([predict], 1))

# y is an array of ONLY the predict values
y = np.array(data[predict])
#The reason why we also have this line outside of th eloop is because even when we dont have to train again we still need these values.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

#This for loop trains over and over and keeps the most accurate training
"""
bestScore = 0
for _ in range(999999):

    # This creates 4 different arrays, X arrays have the predictions and Ys have the actual data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


    # The computer will create a linear function in order to plug in data and get an output
    linear = linear_model.LinearRegression()


    #Trys to find a best fit line for all of the data
    linear.fit(x_train, y_train)

    accuracy = linear.score(x_test, y_test)
    print("Accuracy: ", accuracy*100 , "%")

    if accuracy > bestScore:
        bestScore = accuracy
        # This saves the training so that I dont have to retrain my computer every single time and if i get a high accuracy I can keep that
        # it saves our model
        print("Creating new pickle file with the accuracy of ", accuracy)
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
print("\n\nFinal Accuracy", bestScore)
"""


# loads in the pre exsisting model from the pickle file
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

#Since this is creating a linear function (y = mx + b), lets Find out what m is and what b is
print("Linear function m value",linear.coef_)
print("Linear function y intercept",linear.intercept_)
#the reason why we have multiple m values is because the linear graph goes in multiple dimensions. On a regular 2d line we only have 2 m values

predictions = linear.predict(x_test);

for x in range(len(predictions)):
    print("Computer Guess: ", predictions[x], "   First term grade, second term grade, hours of study, failures, absences: ", x_test[x], "   Actual grade:", y_test[x])

#mkaes our grid look decent on matlib
style.use("ggplot")
p = "G1"
    # add your x and y value, Y will be what you are trying to predict"
pyplot.scatter(data[p], data["G3"])
#Set up the labels
pyplot.xlabel(p)
pyplot.ylabel("Final Grades")

#show the graph
pyplot.show()