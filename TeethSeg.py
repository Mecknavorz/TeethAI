#teeth segmentation AI
#Made by Tzara Northcut (@Mecknavorz)
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

'''
load model config
'''
tf.keras.models.model_from_config("/home/tzara/SeniorDesign/TeethAI/teet_seg.config", custom_objects=None)
