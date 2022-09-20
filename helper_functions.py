import os
import pathlib
import datetime
import zipfile
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import  matplotlib.image as mpimg
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def view_random_image(directory_path,class_name):
  filepath = directory_path+"/"+class_name+"/"
  print(filepath)
  random_image = random.sample(os.listdir(filepath),1)
  img = mpimg.imread(filepath+"/"+random_image[0])
  plt.imshow(img)
  plt.title(class_name)
  plt.axis("off")
  return img

def create_model(model_url,num_classes,image_shape):
  """Takes a model url from tensorflow hub and build a model with it"""
  feature_extractor_layer = hub.KerasLayer(model_url,trainable=False,name="feature_extraction_layer",input_shape=image_shape+(3,))

  model = tf.keras.Sequential([
      feature_extractor_layer,
      layers.Dense(num_classes,activation="softmax",name="output_layer")
  ])
  return model


def plot_loss_curves(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"]))

    #plot loss
    plt.plot(epochs,loss,label="training loss")
    plt.plot(epochs,val_loss,label="Val loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()

    #plot accuracy
    plt.figure()
    plt.plot(epochs,accuracy,label="training accuracy")
    plt.plot(epochs,val_accuracy,label="Val Accuracy")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()
    
    
def original_vs_augmented(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)
  
def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name +"/"+ experiment_name + datetime.datetime.now().strftime("%Y$m$d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=0,write_graph=True,update_freq='epoch')
  print(f"Creating tensorboard log file in {log_dir}")
  return tensorboard_callback    
  
def preprocess_and_plot_prediction(filename,model,class_names):
  image = tf.io.read_file(filename)
  image = tf.image.decode_image(image)
  image = tf.image.resize(image,size=(224,224))
  image = image/255.
  image = tf.expand_dims(image,axis=0)
  predictions = model.predict(image)
  predicted_label = class_names[np.argmax(predictions)]
  plt.figure(figsize=(10,7))
  plt.imshow(tf.squeeze(image))
  plt.title(f"Predicted Label: {predicted_label}")
  plt.axis("off")
  
def plot_truth_and_predictions(images,true_labels,class_names,model):
    """Plot 30 images and show their truth labels and predicted labels:
       images: images to plot
       true_labels: The truth labels of the images
       class_names: The class_names of the labels
       model: model to use for prediction 
       Works only for binary classification problem
    """
    predictions = np.around(model.predict(images))
    predictions  
    plt.figure(figsize=(20,15))
    for i in range(30):
      plt.subplot(10,3,i+1)
      plt.imshow(images[i])
      plt.title(f"Truth:{class_names[true_labels[i]]} Predicted:{class_names[int(predictions[i][0])]}")
      plt.axis("off")
    plt.tight_layout()    
