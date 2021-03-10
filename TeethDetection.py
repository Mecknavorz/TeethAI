#basic AI to just give a yes or no if teeth are in the picture
#made by Tzara Northcut (@Mecknavorz)
import tensorflow as tf

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
