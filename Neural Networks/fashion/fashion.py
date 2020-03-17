import tensorflow as tf
from tensorflow import keras # helps us get data sets
import numpy as np # allows us to have arrays

data = keras.datasets.fashion_mnist # loads in dataset

# split the data into trainig and testing
(train_images, train_labels), (test_images, test_labels) = data.load_data()
# each image has 784 values, because it is a 28x28 pixel image


# Our lists of data have values between 0 and 255, If we divide them all by 255, they will all be values between 0 and 1
train_images = train_images/255.0
test_images = test_images/255

# our labels are values between 0 and 9 where each value represents a type of clothing. (ex. 0 is a T shirt, 8 is a bag)
# so lets add these to a list so when the computer tries to tell us that the image is an 8 for example, it will print 'Bag'
class_names = [
    'T Shirt',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandle',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle Boot'
]

# creating model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # taking in the input
    keras.layers.Dense(128, activation='relu'), # Makes a layer with 128 neurons
    keras.layers.Dense(10, activation="softmax") # last layers with 10 neurons (for our 10 pieces of clothing)
])

# compile model (picking optimizer, loss function, and metrics to keep track of)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# training model
model.fit(train_images, train_labels, epochs=7)

# testing the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy: ', test_acc)