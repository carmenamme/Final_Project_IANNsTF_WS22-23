# -*- coding: utf-8 -*-
'''File that initializes the pretrained ResNet50 from the Keras library.
   
   Part of the Final Project for the Module "Implementing IANNs with TensorFlow" Winter Term 2022/2023
   Final Project by Carmen Amme (994813) and Anneke Büürma (995025)
'''

# import libraries 
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers
from tensorflow.python.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

class Pretrained_ResNet50():
  ''' Class that initializes the pre-trained ResNet50 model from the keras library. 
      Holds certain functionalities
  '''
  
  def __init__(self, **kwargs):
        ''' The init method initializes the pre-trained model with its pre-trained weights '''
        super().__init__(**kwargs)

        self.base_model = ResNet50(weights='imagenet')
        self.base_model.trainable = False

  def modelSummary(self):
        ''' This function retuns the model summary. '''
        return self.base_model.summary()

  def getLayerOut(self, layer_name, data):
        ''' This function builds a model that outputs the activation of a certain layer. A new model is built on based on the 
            given model and the layer which defines the output nodes of the model and therefore returns the activation of that model.
  
            Parameters:
                  layer_name: String that holds the name of the target layer
                  data: an image that is put through the built model 

            Returns:
                  The layer activation 
        '''
        
        layer_output = self.base_model.get_layer(layer_name).output
        model = Model(self.base_model.input, outputs=layer_output)
        result = model.predict(data)
        #print(result)
        return result

pretrainedResnet50 = Pretrained_ResNet50()