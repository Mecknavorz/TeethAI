#convert our saved object detection file to a tflite file
#so we can use it in the app
#Made by Tzara Northcut (@Mecknavorz)
import tensorflow as tf

#keras_model = model.keras_model
#load the model
print("loading model...", end="")
keras_model = tf.saved_model.load("/home/tzara/SeniorDesign/TeethAI/training_demo/exported-models/teeth_seg/saved_model/")
print("Done!")

print("converting model...", end="")
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.alllow_custom_ops=True
converter.post_training_quantize =True
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.experimental_new_converter = True

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
print("Done!")

print("Writing to file...", end="")
open("teeth_seg.tflite", "wb").write(tflite_model)
print("Done!")
