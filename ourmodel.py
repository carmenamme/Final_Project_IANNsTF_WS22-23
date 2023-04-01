# -*- coding: utf-8 -*-
'''Class to easily read in our model with weights to do further analysis.

   Part of the Final Project for the Module "Implementing IANNs with TensorFlow" Winter Term 2022/2023
   Final Project by Carmen Amme (994813) and Anneke Büürma (995025)
'''

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.models import Model

class OurModel():
    """Class that initializes our model.
    Holds certain functionalities
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        initializer = tf.keras.initializers.Zeros()

        self.model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(input_shape=(43, 43, 3), filters=43, kernel_size=3, padding='same', activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Conv2D(filters=43, kernel_size=3, padding='same', activation='relu', kernel_initializer=initializer),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

        tf.keras.layers.Conv2D(filters=86, kernel_size=3, padding='same', activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Conv2D(filters=86, kernel_size=3, padding='same', activation='relu', kernel_initializer=initializer),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

        tf.keras.layers.Conv2D(filters=172, kernel_size=3, padding='same', activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Conv2D(filters=172, kernel_size=3, padding='same', activation='relu', kernel_initializer=initializer),
        tf.keras.layers.GlobalAvgPool2D(),

        tf.keras.layers.Dense(91, activation='sigmoid', kernel_initializer=initializer)])
    
    def use_trained_weights(self, trained_model):
        '''
        Uses the weights of our trained model to put them in the newly set-up model with the same architecture to later extract layer activations.

        Parameters:
            trained_model: the trained model as read in with preprocessing.load_trained_model()
        
        Returns:
            The model with weights.
        '''
        trained_layers = [i.name for i in trained_model.layers]
        new_layers = [i.name for i in self.model.layers]

        for i in range(len(trained_layers)):
            trained_layer_weights = trained_model.get_layer(trained_layers[i]).weights
            self.model.get_layer(new_layers[i]).set_weights(trained_layer_weights)
        
        return self.model


    def predictions(self, data):
            # get predictions
            preds = self.model.predict(data)
            print('Predicted:', decode_predictions(preds, top=3)[0])
            return preds

    def modelSummary(self):
            ''' This function retuns the model summary. '''
            return self.model.summary()
    

    def get_layer_weights(self, layer):
         '''Returns the weights for a given layer.
         '''
         return self.model.get_layer(layer).weights


    def getLayerOut(self, layer_name, data):
        """
        getLayerOut function that gives the layer output of a certain layer

        Parameters:
            layer_name: layer of interest
            data: one image that will be shown to the model
        
        Returns:
            Layer output of the layer of interest regarding the data input.
        """
        
        layer_output = self.model.get_layer(layer_name).output
        model = Model(self.model.input, outputs=layer_output)
        result = model.predict(data)
        return result