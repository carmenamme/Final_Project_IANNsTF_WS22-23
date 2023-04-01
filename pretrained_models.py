# -*- coding: utf-8 -*-
'''Functions that deal with images and layer activations
   
   Part of the Final Project for the Module "Implementing IANNs with TensorFlow" Winter Term 2022/2023
   Final Project by Carmen Amme (994813) and Anneke Büürma (995025)
'''

import os
from os import listdir
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers
from tensorflow.python.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


def getActivations(model, layerName, folderDir, order_special100, target_size):
  ''' This function calculates a certain layer output of a defined model for each image in the dataset
    
      Parameters:
        model: tensorflow keras model (either pre-trained or self-trained) 
        layerName: String with name of the target layer
        folderDir: String with the directory of the special100 dataset
        order_special100: order in which the special100 images were shown to the participant

      Returns:
        An array with the layer responses for each image
  '''

  counter = 0
  
  #make sure that we have the same order for fMRI images and model images to compare RDMs
  for image_id in order_special100: 
    for image_name in os.listdir(folderDir):
      if (image_name.endswith('.png')):
        split = image_name.replace('d', '.').split('.')
        id = split[-2]
        if str(id) == str(image_id):
          inp_img = generate_inp_image(os.path.join(folderDir, image_name), target_size=target_size)
          out = model.getLayerOut(layerName, inp_img).flatten()
          if(counter == 0):
            layer_responses = out
          elif(counter != 0):
            layer_responses = np.vstack((layer_responses, out))
          counter += 1

  return layer_responses.transpose()


def generate_inp_image(img_path, target_size):
  ''' This function loads and processes the images one by one so that they can be fed into the model
  
      Parameters:
        img_path: String that holds the path of a certain image

      Returns:
        An processed image that can be fed into a model
  '''

  # get path of the images
  #img_path = '/content/gdrive/MyDrive/NSD_Special100/shared0008_nsd03172.png'
  img = image.load_img(img_path, target_size=target_size)
  inp_img = image.img_to_array(img)
  inp_img = np.expand_dims(inp_img, axis=0)
  inp_img = preprocess_input(inp_img)
  return inp_img