# TeethAI
Teeth AI for senior design project

## Main Directory 
Some Important Files in the Main directory include:

#### FileStuff.py
a small python file with little commands to help edit files for managing the dataset and other things. The most important functions in it are as follows:
- `makeboxes(filepath)`: **this the function that turns an image at a file path into a mask w/bounding boxes (when fully un-commented) or returns the values for the boudnig boxes of the masks by default**
- `make_tfrecord(record_name, masks, images)`: this function is used to turn the datasets into `.tfrecord` files that the object detection API a can actually use

#### TeethDetection.py
A binary classifier originally made to test out TensorFlow functions. Ideally it tells you whether or not teeth are visible in an image, results were shoddy.

#### TeethSeg.py
This is the file used to turn images in [`demo_images`](/training_demo/demo_images) into annotated images w/bounding boxes, if the faster R-CNN worked properly this would produce a comprehensible output but unfortunately it mostly returns gibberish. Some code to make note of in this file is:
- `max_boxes_to_draw`: which decides the maximum number of boxes the AI is allowed to guess are there
- `min_score_thresh`: which is the minimum score a prediction box is allowed to have to be drawn
Tampering with either of these settings will allow for more fine turning of the output, if the faster R-CNN actually worked in the first place.

#### run.txt
This file contains the commands which need to be run (ideally) in a virtual environment to actually preform training and testing operations on our TensorFlow Model


## dataset
This Folder contains all of the images used for training both AIs used in this project, most of them are from the [ODSI-DB](https://sites.uef.fi/spectral/odsi-db/), while the remain were provided by our class sponsor.


## training_demo
This Folder contains all the environment stuff needed to custom train TensorFlow Object Detection API Models. As it stands the only one currently downloaded and expirmented on is the faster R-CNN. It is set up in such a way that it would not be difficult to custom train other models as well. The faster R-CNN as well as other pre-trained models can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). The significance of each of the folders is as follows
- [`annotations`](training_demo/annotations): contains the `labels.pbtxt` which is a file telling us information about the classes we're trying to detect
- [`demo_images`](training_demo/demo_images): contains the images used to test AI output, mainly used by `TeethSeg.py`
- [`exported_models`](training_demo/exported-models): Is where we save our model, as it stands `teeth_seg` is the only model there, but if other pre-trained models were downloaded other saved models would be there as well
- [`images`](training_demo/images): contains a sorted version of the dataset from [`dataset`](main/dataset) in the main directory. It is divided into testing and training folders each containing image for testing and training respectively.
- [`models`](trainin_demo/models): contains folders with the models for the AIs we train off of and **most importantly it contains the customized** `pipeline.config`
  -`pipeline.config`: Controls basically all of the aspects of training from batch size, to number of training epochs to location of the datasets and tfrecords. Every model that we custom train needs to have their own specialized version of this file.
- [`pre-trained-models`](training_demo/pre-trained-model): contains the various pretrained models we want to custom train on. Models can be downloaded from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
It also contains the two programs which we run in a virtual environment in order to actually train and export the AI, for more information look at `run.txt` in main.


## UNet
UNet is the AI found by [Joseph Norman](https://github.com/josephnormandev), and created by Olaf Ronneberger at the Computer Science Department of the University of Freiburg, Germany, for more information on how UNet works [check out the blogpost here](https://idiotdeveloper.com/polyp-segmentation-using-unet-in-tensorflow-2/).
