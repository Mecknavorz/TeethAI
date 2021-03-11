#basic AI to just give a yes or no if teeth are in the picture
#made by Tzara Northcut (@Mecknavorz)
#impors and stuff
import tensorflow as tf
import numpy as np
import os #for system calls
import cv2 #the convolutional netwrok libary iirc CHECK THIS
#these two are for help with the dataset
import tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.preprocessing import image
#for plotting and giving us visual feedback w/the training process
import matplotlib.pyplot as plt

"""
dataset setup
"""
#NEED TO DO THIS THURSDAY FOR SURE!


"""
Initial model set up
"""
#the inital model
model = tf.keras.models.Sequential()
#layer 1
model.add(keras.layers.Conv2d(16, (3,3), activation='relu', input_shape=(200,200,3)))
model.add(keras.layers.MaxPooling2D(2,2)) #add a max pooling layer to half image dimensions
#layer 2
model.add(keras.layers.Conv2d(32, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
#layer 3
model.add(keras.layers.Conv2d(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
#layer 4
model.add(keras.layers.Conv2d(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
#layer 5
model.add(keras.layers.Conv2d(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
#flatter the layers
model.add(keras.layers.Flattern())
#the hidden layer
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

#compile the model with the binary cross entropy loss since it's binary classification
model.compile(loss='binary_crossentropy' optimizer=RMSprep(lr=0.001), metrics='accuracy')

"""
training and validation stuff
"""
#actualy train the model
#steps per epoch should be the # of training images divided by batch size
mode.fit_generator(training_set, steps_per_epoch=250, epocs=10, validation_set = test_set)

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
    elif guess = 0:
        plt.xlabel("No teeth! :c :c", fontsize=30)
