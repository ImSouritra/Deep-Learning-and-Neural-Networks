import os
import pathlib
import shutil
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


def unzip(filepath):
  """
  Unzip a zipfile
  """
  file_ref = zipfile.ZipFile(filepath)
  print("Unzipping zipfile for ya!!")
  file_ref.extractall()
  file_ref.close()
  
def walk_through_dir(file_path):
  """
  Walk through directory and print what is underlying
  """
  for dir_path, dir_name , filename in os.walk("file_path"):
        print(f"There are {len(dir_path)} directories and {len(filename)} images in {dir_path}") 



def view_random_image(directory_path,class_name):
  """
  View random image from a given directory path and a class name
  """
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
    """Plot loss curves of a model from its history"""
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
  """Show original vs augmented photo side by side"""
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
  """" Preaprocess a given image and plot the image with its predicted label"""
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
    
def make_train_test_split(root_dir,class_names):   
    """" Split your data into train and test directory
    args:
        root_dir: The name of the root directory
        class_names: A list of class names
        
    """
    root_dir = root_dir
    classes = class_names

    for cls in classes:
      os.makedirs(root_dir +'train/' + cls)
      os.makedirs(root_dir +'val/' + cls)
      os.makedirs(root_dir +'test/' + cls)
    
    ## creating partition of the data after shuffeling

    for cls in classes:
      src = root_dir + cls # folder to copy images from
      print(src)

      allFileNames = os.listdir(src)
      np.random.shuffle(allFileNames)

      ## here 0.75 = training ratio , (0.95-0.75) = validation ratio , (1-0.95) = test ratio 
      ##training ratio  
      train_FileNames,val_FileNames,test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)*0.75),int(len(allFileNames)*0.95)])

      # #Converting file names from array to list

      train_FileNames = [src+'/'+ name for name in train_FileNames]
      val_FileNames = [src+'/' + name for name in val_FileNames]
      test_FileNames = [src+'/' + name for name in test_FileNames]

      print('Total images  : '+ cls + ' ' +str(len(allFileNames)))
      print('Training : '+ cls + ' '+str(len(train_FileNames)))
      print('Validation : '+ cls + ' ' +str(len(val_FileNames)))
      print('Testing : '+ cls + ' '+str(len(test_FileNames)))

      ## Copy pasting images to target directory
      
      
def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.
    
    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
    """
    
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()      
      

      for name in train_FileNames:
          shutil.copy(name, root_dir + 'train/'+cls )


      for name in val_FileNames:
          shutil.copy(name, root_dir +'val/'+cls )


      for name in test_FileNames:
          shutil.copy(name,root_dir + 'test/'+cls )
