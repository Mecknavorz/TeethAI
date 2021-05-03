#teeth segmentation AI
#Made by Tzara Northcut (@Mecknavorz)
#based on the paper:
#"A Deep Learning Apporach to Automatic Teeth Detection and Numbering Based on Object Detection in dental Periapical Films"
# by Hu Chen, Kailai Zhang, Peijun Lyu, Hong Li, Ludan Zhang, Ji Wu & Chin-Hui Lee
#if something about this AI doesn't make sense try checking that paper first for explaniations

'''
imports and stuff
'''
#file stuff
import os
os.environ['TF-CPP_MIN_LOG_LEVEL'] = '2' #basically a way to cut out errors that aren't important (probably)
import pathlib
import time
import warnings
warnings.filterwarnings("ignore")

#AI stuff
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR') #more stuff to ignore small errors
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

#image stuff
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

'''
load model and config
'''
#stuff to make sure the GPUs get used, allowing us to do things better faster stronger etc
#(namely memory usage stuff)
#gpus = tf.config.list_logical_devices("GPU")
#for gpu in gpus:
    #tf.config.experimental.set_memory_growth(gpu, True)

#this path needs to be set to the images we want to predict things with
#demo_images is a folder I made within the AI's environment to put some sample images to test on
#might remove this path and have it input by function
IMAGE_PATHS = os.path.join("/home/tzara/SeniorDesign/TeethAI/training_demo/demo_images")
#set up the labels
PATH_TO_LABELS = os.path.join(os.path.dirname(__file__), "/training_demo/annotations/label_map.pbtxt")

#actual stuff to load the model
print("Loading model...", end="")
start_time = time.time()
#load saved model and build the detection function
#FOR WHATEVER CTHULHU FORSAKEN REASON, THIS PICKY BITCH WILL ACCEPT NOTHING LESS THAN THE ABSOLUTE FILE PATH
#WHY? I DON"T KNOW! BUT THAT"S THE WAY THE COOKIE CRUMBLES I GUESS
detect_fn = tf.saved_model.load("/home/tzara/SeniorDesign/TeethAI/training_demo/exported-models/teeth_seg2/saved_model/")
#print out how long it took to load
end_time = time.time()
elapsed_time = end_time - start_time
print("Done! Took {} seconds.".format(elapsed_time))

#load the label map
category_index = label_map_util.create_category_index_from_labelmap("/home/tzara/SeniorDesign/TeethAI/training_demo/annotations/label_map.pbtxt", use_display_name=True)

'''
predicion code
'''
#just turn an image into a numpy array
def img2numpy(path):
    return np.array(Image.open(path))

#for image_path in IMAGE_PATHS: #this was the original code but it doesn't work
IM2 = os.listdir(IMAGE_PATHS)
for im2 in IM2:
    image_path = os.path.join(IMAGE_PATHS, im2)
    print("Runninger interference for {}...".format(image_path), end="")
    image_np = img2numpy(image_path)
    #input needs to be a tensor
    input_tensor = tf.convert_to_tensor(image_np)
    #the model needs a batch so we add a new axis
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections["num_detections"] = num_detections

    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections["detection_boxes"],
        detections["detection_classes"],
        detections["detection_scores"],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=30, #probably crank this down
        min_score_thresh=.20, #also tamper with this if it's too high/low
        agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    print('Done')
    plt.show()


'''
save the model so we can use it in the app
MIGHT NEED TO MOVE THIS TO A NEW FILE
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
