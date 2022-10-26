import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)
# only 10K more popular words

'''Words are coded in numbers, so each word is represented by a number in th order they appear.
To be able to decode them we need to map them. In real projects you have to map them but tensorflow does that for us.'''

# A dictionary mapping words to an integer index
_word_index = data.get_word_index()

word_index = {k: (v + 3) for k, v in _word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# this function will return the decoded (human readable) reviews.

# our test_data[0], test_data[1],etc has no the same length and we need it to be able to teach how to do it.
# use keras function to PAD or remove and get top words at 250.

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post',
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=250)
# we need to preprocess it  in order to be able to use our model.

# model
'''
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
# (words, dimensions). Embedding layers create word-vectors in this case, in 16 dimensions.
model.add(keras.layers.GlobalAveragePooling1D())
# this takes the data and put it in a lower dimension.
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()  # prints a summary of the model

model.compile(optimizer ='adam', loss='binary_crossentropy', metrics=['accuracy'])

# split data in 2 sets, validation data and data

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val),verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

model.save("model.h5")
'''
model = keras.models.load_model('model.h5')


# noinspection PyInconsistentIndentation
def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded


with open("test.txt", encoding="utf-8") as f:
    for line in f.readlines():
        newline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
        encode = review_encode(newline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",
                                                            maxlen=250)  # make the data 250 words long
        predict = model.predict(encode)
        print(predict[0])

'''
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: "+str(predict[0]))
print("Actual: "+ str(test_labels[0]))
'''
