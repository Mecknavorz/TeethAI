#teeth segmentation AI
#Made by Tzara Northcut (@Mecknavorz)
#based on the paper:
#"A Deep Learning Apporach to Automatic Teeth Detection and Numbering Based on Object Detection in dental Periapical Films"
# by Hu Chen, Kailai Zhang, Peijun Lyu, Hong Li, Ludan Zhang, Ji Wu & Chin-Hui Lee
#if something about this AI doesn't make sense try checking that paper first for explaniations

'''
imports and stuff
'''
#the basics
import tensorflow as tf
print(tf.__version__) #print the version we're using rn, might be helpful, can't hurt to have it
print("Available GPU devices: %" % tf.test.gpu_device_name())
import fensorflow_hub as hub #this is where we can easily get the faster R-CNN
import numpy as np #helps with image processing
import time

#seed setting stuff
mlseed = 666 #set the seed
from numpy.random import seed
np.random.seed(mlseed)
import random as rn
rn.seed(mlseed)
import os #for system calls
os.environ['PYTHONHASHSEED']=str(mlseed) #more seed control

#for downlading the images
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesID

#for drawing on the image
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps



'''
Visualization code
'''
def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(false)
    plt.imshow(image)

#probably rework this to pull from the files instead of the database
def download_and_resize(url, new_width=256, new_height=256, display=False):
    _, filename = tempfile.mkstemp(suffix=".png")
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    print("Image downloaded to %s." % filename)
    if display:
        display_image(pil_image)
    return filename

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()):
  #Adds a bounding box to an image.
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, ill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
    draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)
    text_bottom -= text_height - 2 * margin

def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  #Overlay labeled boxes on an image with formatted scores and label names
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 25)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image


'''
actually use the module
'''
#don't know if I'll need this part, we'll see I guess
# By Heiko Gorski, Source: https://commons.wikimedia.org/wiki/File:Naxos_Taverna.jpg
image_url = "https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg" 
downloaded_image_path = download_and_resize_image(image_url, 1280, 856, True)

module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def run_detector(detector, path):
  img = load_img(path)
  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()
  result = {key:value.numpy() for key,value in result.items()}

  print("Found %d objects." % len(result["detection_scores"]))
  print("Inference time: ", end_time-start_time)

  image_with_boxes = draw_boxes(img.numpy(), result["detection_boxes"], result["detection_class_entities"], result["detection_scores"])
  display_image(image_with_boxes)

#run_detector(detector, downloaded_image_path)



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
