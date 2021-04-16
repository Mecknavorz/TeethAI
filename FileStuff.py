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
        cv2.rectangle(output, (x,y), (x+w, y+h), (0, 0, 255), 2)
    # Display the output image
    #b,g,r = cv2.split(output)
    #frame_rgb = cv2.merge((r,g,b))
    #plt.imshow(frame_rgb)
    #plt.title('Boxes! :D :D')
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.show()

#tfrecord maker
#def make_record(target, dest):
    
            
