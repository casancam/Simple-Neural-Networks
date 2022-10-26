import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# images are in a scale of 0 to 255. since this is a large number we can divide the numpy array by 255 to simplify.

train_images = train_images / 255.0
test_images = test_images / 255.0


'''In every image there are 28x28 = 784 pixels (inputs) and out neural network has 10 labels 
(outputs), so we could only tell the computer, if you see this pixels in this color assign this to the 
highest probability. But, in order to make the network more complex we want to add a hidden layer in between 
the input and the output. This hidden layer is going to have 15-20% (in this case 128) of the sample neurons.
By implementing it, we expect the computer to learn and find patterns in order to increase its accuracy.'''

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
    ])
''' First, we flatten the data (2D or 3D data has to be flatten in order to pass it to the neuron as a number 
instead of passing a list. Second line creates a Dense layer. A Dense layer is a layer where all neurones are 
connected between them (hidden layer), last layer are our labels. Using softmax adds numbers from all neurons 
to 1, is like saying: I think this image has X% probability of being this neurone.'''

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
# quite common optimizer and loss

model.fit(train_images, train_labels, epochs=5)
# epochs is how many times the computer is going to see your train data and go through it in different orders.

# you have to pass on a list or np array because it does a bunch of predictions and therefore expects a list.
prediction = model.predict(test_images)

plt.figure(figsize=(5, 5))
for i in range(10):
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.xlabel(class_names[test_labels[i]])
    plt.title(class_names[np.argmax(prediction[i])])
    plt.show()
