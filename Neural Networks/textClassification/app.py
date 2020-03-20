# this app will look at movie reviews and determine if they are possitive or negative

import tensorflow as tf
from tensorflow import keras
import numpy

# = = = Loading in data = = = 
imdb = keras.datasets.imdb # imdb is the movie database, we want to bring in that data

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=88000) #loads data into 4 lists. The data is the most common 10,000 words

# so reviews are written in words, but we cant pass words into a neural network so this data that we loaded are lists of possitive integers that represent words
# Since we want to be able to read these as humans, we need to turn these integers into words
_word_index = imdb.get_word_index() # gives us a dictionary of words and their number. for example 'enjoyment: 3126'

word_index = {k:(v+3) for k,v in _word_index.items()} # this increases the integer value of each word. For example 'enjoyment: 3129'
# so now the integers 0, 1, 2, 3 dont have any words assosiated with them (0 was always open since the dictionary started at 1)

word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2 # unkown
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value,key) for (key, value) in word_index.items()]) # what used to be 'enjoymen: 3129' is now '3129: enjoyment'

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text]) #returns the integer values in a string of words


# Each review has to be the same length in order for us to put it in a neural network. We will set each review length to a length of 250
# so if it is longer than 250, we will trim it. If it is shorter than 250 we will add padding (padding will have a neautal effect on weather the review is possitive or negative)
# keras already has a function for that already so we can pad/trim our training and testing data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'], padding='post', maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'], padding='post', maxlen=250)

'''
# = = = define model = = = 
model = keras.Sequential()
model.add(keras.layers.Embedding(88000,16)) # word embedding is when they try to find the meaning of words. So similar words are categorized together. 'good', 'great', 'amazing' should be all together and 'bad', 'horrible' should be a completely seperated category
model.add(keras.layers.GlobalAveragePooling1D()) # I think this scales down our data to one dimension
model.add(keras.layers.Dense(16, activation='relu')) # this layer is just trying to find paterns with the relu function
model.add(keras.layers.Dense(1, activation='sigmoid')) # one output neuron because we want to know if the review is Possitive or negative, 0 or 1, and the sigmoid funtion is good for that

model.summary()

# compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# we have to split up our data into testing and validating so that we can train the model with some data and test the accuracy with different data
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# now we have to train the model
fitModel = model.fit(
    x_train,
    y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

results = model.evaluate(test_data, test_labels)
print(results)

# Save model
model.save('model.h5')
'''
model = keras.models.load_model('model.h5')

def removeUnwantedCharacters(_string, chars):
    newString = _string
    for char in chars:
        newString = newString.replace(char, "")
    return newString

def review_encode(s):
    encoded = [1] # creates a list and adds a first value of 1 (for start)

    for word in s:
        if word.lower() in word_index: #checks if the word is in the index
            encoded.append(word_index[word.lower()]) #adds the integer value of that word
        else:
            encoded.append(2) # adds unkown to the list
    return encoded

def print_result(review, prediction):
    print('\n\n\n\nYour Review:', review)
    if prediction >= 0.5:
        print('\nI am ' + str(int(prediction * 100)) + '% sure that this is a positive review\n\n')
    else:
        print('\nI am ' + str(round(100 - (float(prediction) * 100), 2)) + '% sure that this is a negative review\n\n')


with open('review.txt') as f:
    unwantedCharacters = [',', '.', '(', ')', ':', '"', '-']
    for line in f.readlines(): # looks at every line in the txtx file incase I have multiple reviews in one file
        nline = removeUnwantedCharacters(line, unwantedCharacters).split(' ') # removes characters out of the review
        encode = review_encode(nline) # turns the review into a list of integers that rep
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index['<PAD>'], padding='post', maxlen=250)
        predict = model.predict(encode)
        print_result(line, predict[0])