#basic AI to just give a yes or no if teeth are in the picture
#made by Tzara Northcut (@Mecknavorz)
#impors and stuff
import tensorflow as tf
import numpy as np #helps with image processing
import os #for system calls
import cv2 #the convolutional netwrok libary iirc CHECK THIS
from tensorflow import keras #used for a lot of the heavy lifting in terms of CNN stuff
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
#these two are for help with the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
#for plotting and giving us visual feedback w/the training process
import matplotlib.pyplot as plt

"""
dataset setup
"""
#rescale the images some
train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

#set up stuff for the training set,
#obv change the link to the destination to where your datasets are stored
train_dataset = train.flow_from_directory("/home/tzara/SeniorDesign/dataset/binary-teeth/train", target_size=(200,200), batch_size=32, class_mode="binary")
test_dataset = test.flow_from_directory("/home/tzara/SeniorDesign/dataset/binary-teeth/test", target_size=(200,200), batch_size=32, class_mode="binary")



"""
Initial model set up
"""
#the inital model
model = keras.Sequential()
#layer 1
model.add(keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200,200,3)))
model.add(keras.layers.MaxPooling2D(2,2)) #add a max pooling layer to half image dimensions
#layer 2
model.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
#layer 3
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
#layer 4
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
#layer 5
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
#flatter the layers
model.add(keras.layers.Flatten())
#the hidden layer
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

#compile the model with the binary cross entropy loss since it's binary classification
model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.RMSprop(lr=0.001), metrics='accuracy')

"""
training and validation stuff
"""
#actualy train the model
#steps per epoch should be the # of training images divided by batch size
#need to double check current values, especially once I remake the dataset
history = model.fit(train_dataset, steps_per_epoch=20, epochs=10, validation_data=test_dataset, validation_steps=8)
#mode.fit_generator(training_set, steps_per_epoch=250, epocs=10, validation_data=test_set)

#evaluate the accuracy of the model
model.evaluate(test_dataset)
STEP_SIZE_TEST = test_dataset.n//test_dataset.batch_size
test_dataset.reset()
prediction = model.predict(test_dataset, verbose=1) #verbose =1 so we can debug n'stuff

#compute false positive and true positive rate so we can figure out the ROC curve
falsepr, truepr = roc_curve(test_dataset.classes, prediction)
roc_auc = auc(falsepr, turepr)
plt.figure()
lw = 2 #since we're constrasting against a constant, we only need a constant
plt.plot(falsepr, truepr, color="red")
#set up the labels for the plot
lw = lw, label="ROC curve (Area = %0.2f)" % roc_auc
#plot the graph
plt.plot([0,1], [0,1], color="blue", lw=lw, linestyle="--")
#bounds for the graph
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
#labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Postive Rate")
#title the graph
plt.title("Receiver Operating Characterist Example")
#give a key
plt.legend(loc="lower right")
#show us the money!
plt.show()

"""
Prediciton stuff
"""
def classify(file):
    #this line might not work and if it doesn't reimplement w/ sys/os commands instead of Keras
    img = image.load_img(file, target_size=(200,200)) #load the image
    plt.imshow(img) #shwo the image
    #set up the axis
    Y = image.img_to_array(img1)
    X = np.expand_dims(Y, axis=0)
    #actually make the predicton
    guess = model.predict(x)
    print(guess) #tell what we think it is
    if guess == 1:
        plt.xlabel("Teeth! :D :D",fontsize=30)
    elif guess == 0:
        plt.xlabel("No teeth! :c :c", fontsize=30)
