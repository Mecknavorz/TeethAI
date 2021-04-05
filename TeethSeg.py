#teeth segmentation AI
#Made by Tzara Northcut (@Mecknavorz)
#based on the paper:
#"A Deep Learning Apporach to Automatic Teeth Detection and Numbering Based on Object Detection in dental Periapical Films"
# by Hu Chen, Kailai Zhang, Peijun Lyu, Hong Li, Ludan Zhang, Ji Wu & Chin-Hui Lee
#if something about this AI doesn't make sense try checking that paper first for explaniations

'''
imports and stuff
'''
#file stuff and misc
import tensorflow as tf
import numpy as np #helps with image processing
import tempfile
import time

#seed setting stuff
mlseed = 666 #set the seed
from numpy.random import seed
np.random.seed(mlseed)
import random as rn
rn.seed(mlseed)
import os #for system calls
os.environ['PYTHONHASHSEED']=str(mlseed) #more seed control

#for most of the AI stuff
import cv2 #the convolutional netwrok libary iirc CHECK THIS
from tensorflow import keras #used for a lot of the heavy lifting in terms of CNN stuff
import tensorflow_hub as hub
from six.moves.urllib.request import urlopen
from six import BytesIO
import scipy.misc
#from sklearn.metrics import roc_curve
#from sklearn.metrics import auc
#these are for the object detection stuff, not sure if I need this
'''
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import colab_utils
from object_detection.utils import model_builder
'''

#these two are for help with the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

#for plotting and giving us visual feedback w/the training process
import matplotlib
import matplotlib.pyplot as plt
%matplotlib linline

#for drawing on the images
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

'''
load model and config
'''
#load the model based off the config the paper used, we can then modify the parameters to match
#I think I have to load the model detection AI in this way
model = tf.keras.models.model_from_config("/home/tzara/SeniorDesign/TeethAI/teet_seg.config", custom_objects=None)
#the AI in the paper might need a second network, leaving this here just in case
#model2 = tf.keras.models.clone_model(model1, input_tensors=None, clone_function=None)


'''
training and validation stuff
'''



'''
save the model so we can use it in the app
'''
#save and convert the file to tflite
def save_and_store(filepath, name):
    #to_export = tf.keras.Model(model)
    file1 = filepath + "/" + name   #for saving the non-lite model
    name2 = name + ".tflite"        #for swaving the converted model
    model.save(file1) #save our current model
    converter =tf.lite.TFLiteConverter.from_saved_model(file1) #load the model
    tflite_model = converter.convert() #actually convert it
    #tflite_model.export(export_dir=filepath)
    #save our new tf lite file
    with open(name2, 'wb') as f:
        f.write(tflite_model)
    #try and generate labels.txt
    #PRETTY SURE THIS PART IS BROKEN
    with open('labels.txt', 'w') as f2:
        for label in labels:
            f2.write(label)
            f2.write('\n')
