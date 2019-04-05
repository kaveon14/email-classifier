from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
 

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

"""## Explore the data

Let's explore the format of the dataset before training the model. The following shows there are 60,000 images in the training set, with each image represented as 28 x 28 pixels:
"""

train_images.shape

"""Likewise, there are 60,000 labels in the training set:"""

len(train_labels)

"""Each label is an integer between 0 and 9:"""

train_labels

"""There are 10,000 images in the test set. Again, each image is represented as 28 x 28 pixels:"""

test_images.shape

"""And the test set contains 10,000 images labels:"""

len(test_labels)

"""## Preprocess the data

The data must be preprocessed before training the network. If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:
"""


"""We scale these values to a range of 0 to 1 before feeding to the neural network model. For this, we divide the values by 255. It's important that the *training set* and the *testing set* are preprocessed in the same way:"""

train_images = train_images / 255.0

test_images = test_images / 255.0


"""## Build the model

Building the neural network requires configuring the layers of the model, then compiling the model.

### Setup the layers

The basic building block of a neural network is the *layer*. Layers extract representations from the data fed into them. And, hopefully, these representations are more meaningful for the problem at hand.

Most of deep learning consists of chaining together simple layers. Most layers, like `tf.keras.layers.Dense`, have parameters that are learned during training.
"""



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=5)

"""As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.88 (or 88%) on the training data.

## Evaluate accuracy

Next, compare how the model performs on the test dataset:
"""

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

"""It turns out, the accuracy on the test dataset is a little less than the accuracy on the training dataset. This gap between training accuracy and test accuracy is an example of *overfitting*. Overfitting is when a machine learning model performs worse on new data than on their training data.

## Make predictions

With the model trained, we can use it to make predictions about some images.
"""

predictions = model.predict(test_images)

"""Here, the model has predicted the label for each image in the testing set. Let's take a look at the first prediction:"""

predictions[0]

"""A prediction is an array of 10 numbers. These describe the "confidence" of the model that the image corresponds to each of the 10 different articles of clothing. We can see which label has the highest confidence value:"""

np.argmax(predictions[0])

"""So the model is most confident that this image is an ankle boot, or `class_names[9]`. And we can check the test label to see this is correct:"""

test_labels[0]

"""We can graph this to look at the full set of 10 channels"""

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


# Grab an image from the test dataset
img = test_images[0]

print(img.shape)

"""`tf.keras` models are optimized to make predictions on a *batch*, or collection, of examples at once. So even though we're using a single image, we need to add it to a list:"""

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

"""Now predict the image:"""

predictions_single = model.predict(img)

print(predictions_single)
 

"""`model.predict` returns a list of lists, one for each image in the batch of data. Grab the predictions for our (only) image in the batch:"""

prediction_result = np.argmax(predictions_single[0])
print(prediction_result)

"""And, as before, the model predicts a label of 9."""