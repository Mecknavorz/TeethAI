#basic functions and stuff for file manipulation
#ideally to help with dataset manipulation
#might not wind up using this file but better safe than sorry
#Made by Tzara Northcut (@Mecknavorz)
import os, fileinput
from PIL import Image
from PIL import ImageOps
import glob #used for iterating through files?
import cv2
#from scipi import ndimage, misc

#using this to unpack the TIFF pictures so I can reoganize
#and give them simple labels
#source is where to open the files from, destination is where to put them
def unpack_tiffs(target, destination):
    tifs = os.listdir(target) #get all the files in the directory
    for f in tifs:
        #stuff to extract the individual photos from the tif
        print(f)
        
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
        save2 = os.path.join(target, "flip_"+file)
        img = Image.open(openImg) #open the file
        img = ImageOps.mirror(img) #mirror
        img.save(save2) #save

#since some of the image files don't have masks generate blank ones for em
#images is the image folder for the dataset
#masks is the folder with the masks which we will save too
#template is the image to copy and rename for out new masks
def fill_gaps(images, masks, template):
    #arays to store the names of the files so we can compare
    #and see what's missing
    masknames = []
    imagenames = []
    #iterate over both folders and compile the lits
    for f in os.listdir(images):
        masknames.append(f)
    for g in os.listdir(masks):
        masknames.append(g)
    #figure out whjat names aren't in the mask file
    for i in masknames:
        if i not in masknames:
            
