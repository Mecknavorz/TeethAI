#basic functions and stuff for file manipulation
#ideally to help with dataset manipulation
#might not wind up using this file but better safe than sorry
#Made by Tzara Northcut (@Mecknavorz)
import os, fileinput
from PIL import Image
from PIL import ImageOps
import glob #used for iterating through files?
import cv2
import matplotlib
import matplotlib.pyplot as plt
from object_detection.utils import dataset_util
import tensorflow.compat.v1 as tf

#from scipi import ndimage, misc

#using this to unpack the TIFF pictures so I can reoganize
#and give them simple labels
#source is where to open the files from, destination is where to put them
def unpack_tiffs(target):
    tifs = os.listdir(target) #get all the files in the directory
    for f in tifs:
        #stuff to extract the individual photos from the tif
        splits = f.split("_", 1)
        nname = splits[0] + ".png"
        print(nname)
        
#mass rename is just to rename all files in a folder sequentially
#give all files the name followed by a number
#target is the target folder, name is the new name
def mass_rename(target, name):
    #files = os.listdir(target)
    #num = 0
    for num, f in enumerate(os.listdir(target)):
        rname = name + "_" + str(num) + ".png"
        rname = target + "/" + rname
        t = target + "/" + f
        os.rename(t, rname)

#mirror all the images
'''
mirror("/home/tzara/SeniorDesign/dataset/binary-teeth/train/no_teeth",
"/home/tzara/SeniorDesign/dataset/binary-teeth2/train/no_teeth")
'''
def mirror(source, target):
    for file in os.listdir(source):
        #establish filepaths to use
        openImg = os.path.join(source, file)
        splits = file.split(".", 1) #unclutter filename
        save2 = os.path.join(target, splits[0]+"m.png")
        img = Image.open(openImg) #open the file
        img = ImageOps.mirror(img) #mirror
        img.save(save2) #save

#unclutter the names
def shorten_names(target):
    #print(target)
    for f in os.listdir(target):
        splits = f.split("_", 1)
        nname = "" #the var we'll use to store new names
        if not splits[0].endswith(".png"):
            nname = splits[0] + ".png"
        else:
            nname = splits[0]
        #print(nname)
        #make the full file paths
        f2 = target + f
        nname2 = target + nname
        #print("target: " + f2)
        #print("new: " + nname2)
        os.rename(f2, nname2)
        

#since some of the image files don't have masks, delete them
#images is the image folder for the dataset
#masks is the folder with the masks which we will use to compare
def fill_gaps(images, masks):
    #arays to store the names of the files so we can compare
    #and see what's missing
    masknames = []
    imagenames = []
    #iterate over both folders and compile the lits
    for f in os.listdir(images):
        imagenames.append(f)
    for g in os.listdir(masks):
        masknames.append(g)
    #figure out what names aren't in the mask file
    for i in imagenames:
        if i not in masknames:
            #delete the file if there is no mask for it
            os.remove(images + i)

'''
THIS MAY BE THE MOST IMPORTANT FUNCTION IN THIS STUPID FUCKING CODE
it turns the masks into bounding boxes, because we should've made those instead
whatever tho, we're here now and there's no point in whining over what could've
worked better.
'''
#make bounding boxes for the teeth
def make_boxes(filepath):
    img = cv2.imread(filepath)
    tbr = [] #array containing the bounding boxes for the blobs
    
    # Threshold the image to extract only objects that are not black
    # You need to use a one channel image, that's why the slice to get the first layer
    tv, thresh = cv2.threshold(img[:,:,0], 1, 255, cv2.THRESH_BINARY)
    
    # Get the contours from your thresholded image
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    # Create a copy of the original image to display the output rectangles
    output = img.copy()
    # Loop through your contours calculating the bounding rectangles and plotting them
    for c in contours:
        x, y, w, h = cv2.boundingRect(c) #this is the only part we really need
        tbr.append([x, y, (x-w), (y-h)])
        #cv2.rectangle(output, (x,y), (x+w, y+h), (0, 0, 255), 2)
    #print(tbr)
    return tbr
    

#tfrecord maker stuff, taken from the models tensorflow repository
#modified to work for our dataset specfically
flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(target, imgBytes, bounding, imgDims):
    height = imgDims[0] # Image height
    width = imgDims[1] # Image width
    filename = target # Filename of the image. Empty if image is not from file
    encoded_image_data = imgBytes # Encoded image bytes
    image_format = b'png' # b'jpeg' or b'png'

    #arrays in which we feed the data into the tfrecord
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for i in bounding:
        #append bound box dimensions to our arrays
        xmaxs.append(i[0])
        ymaxs.append(i[1])
        xmins.append(i[2])
        ymins.append(i[3])
        #append stuff for class and name
        classes_text.append("tooth")
        classes.append(1)
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

#might need an output folder?
def make_tfrecord(masks, images):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    #go over each of the files and figure out what we need 
    for example in os.listdir():
        '''since the mask and the pictures have the same names
        we only know the name of one in order to operate on both
        it's simply a matter of adjusting the path (which are
        provided as the input'''
        #make bounding boxes for our image based off the mask
        maskpath = os.path.join(masks, example)
        bounding = make_boxes(maskpath)
        #make the byte data based off the actual image not the mask
        impath = os.path.join(images, example)
        imgBytes = []
        with open(impath, "rb") as image:
            f = image.read()
            imgBytes = bytearray(f)
            #ALSO GET THE IMAGE DIMENSIONS HERE!!!!!!

        
        #I need to checkif the w/h we feed in is what we start with or scale to
        tf_example = create_tf_example(example, imgBytes, bounding, imgDims)
        writer.write(tf_example.SerializeToString())
    writer.close()
