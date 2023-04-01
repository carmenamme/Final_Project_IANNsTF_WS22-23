# -*- coding: utf-8 -*-
''' Script for the self-trained CNN:
    Part of the Final Project for the Module "Implementing IANNs with TensorFlow" Winter Term 2022/2023 <br >
    Final Project by Carmen Amme (994813) and Anneke Büürma (995025)

    Link to the ms-coco data with resolution 43x43x3 with corresponding multi-hot targets (which of the 91 objects are in the image) 
    and clip embeddings. Around 1 GB.

    https://drive.google.com/file/d/1qw8MUwGxm8Yz899_SnnCSsMHVby-CYl-/view?usp=sharing

    The dataset has 2 groups ("train" and "val") with different datasets which keep the images, the coco ids, the clip embeddings 
    and the labels. With the help of the *keys()* method, the concrete names can be found.

    Installations and imports of packages and toolboxes.
    Further, connect to the OneDrive that keeps all the datasets and necessary logs.
'''

# Commented out IPython magic to ensure Python compatibility.
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.random.set_seed(0)
from tensorflow import keras
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import datetime
from google.colab import drive
import h5py

# Load the TensorBoard notebook extension
# %load_ext tensorboard

# mount a google drive folder to save and load data and models
drive.mount('/content/drive')

'''Read and save the data from the hdf5 file
   Generator to read in hdf5 file

    https://stackoverflow.com/questions/48309631/tensorflow-tf-data-dataset-reading-large-hdf5-files
    https://machinelearningmastery.com/a-gentle-introduction-to-tensorflow-data-api/
'''

#hdf5 Sequence generator
class HDF5Sequence(tf.keras.utils.Sequence):
    '''
        Generator class that ecodes the ID's, the images and the multi-hot targets from the hdf5 file.

    '''

    def __init__(self, hdf5_file, dataset_name, shuffle=True):
        self.hdf5_file = h5py.File(hdf5_file, 'r')
        self.dataset = self.hdf5_file[dataset_name]
        self.shuffle = shuffle
        self.indices = np.arange(self.__len__())
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            
    def __len__(self):
        return len(self.dataset["images"])
 
    def __getitem__(self, idx):
        
        data = {}
        
        idx = self.indices[idx]
        data["idx"] = idx
        img = self.dataset["images"][idx]
        tar = self.dataset["img_multi_hot"][idx]
        data["image"] = img
        data["target"] = tar
        

        # shuffle indices once all images have been read
        if idx == self.__len__()-1:
            self.on_epoch_end()
        
        return data
   
    def on_epoch_end(self):
 
        #'Updates indices after each epoch'
        self.indices = np.arange(0,self.__len__())
        if self.shuffle == True:
            np.random.shuffle(self.indices)

# save path of the hdf5 dataset
hdf5_path = '/content/drive/MyDrive/ms_coco_43.h5'

# decode the data of the hdf5 file into a sequence
hdf5_file = HDF5Sequence(hdf5_path, 'train')

# signature to put the data into a certain structure
output_signature=({"idx": tf.TensorSpec((),dtype=tf.int32),
                       "image": tf.TensorSpec((43,43,3), dtype=tf.uint8),
                        "target": tf.TensorSpec((91,), dtype=tf.uint8)})

# get the data from the generator
ds = tf.data.Dataset.from_generator(lambda: hdf5_file, output_signature=output_signature)

# save image and target into seperate values - preparation to build tuples
first_iter = True

for element in ds:
  image = [element["image"]]
  target = [element["target"]]
  
  if first_iter:
    img_arr = image
    tar_arr = target
    first_iter = False
  else:
    img_arr = np.vstack((img_arr, image))
    tar_arr = np.vstack((tar_arr, target))
    if len(img_arr) % 100:
      print(len(img_arr))

# build a tensorflow dataset from the values
train_dataset = tf.data.Dataset.from_tensor_slices((img_arr, tar_arr))
test_dataset = tf.data.Dataset.from_tensor_slices((img_arr, tar_arr))

# save the train and test data into a predefined path into google drive
train_path = '/content/drive/MyDrive/model_data/train'
tf.data.Dataset.save(train_dataset, train_path)

test_path = '/content/drive/MyDrive/model_data/test'
tf.data.Dataset.save(test_dataset, test_path)

'''--------------------------------------------------------------------'''

'''Load the built datasets from the paths in google drive we will provide you these datasets -> please adjust the paths'''

# define paths where the datasets have been saved and load them
train_path = '/content/drive/MyDrive/model_data/train'
test_path = '/content/drive/MyDrive/model_data/test'

train_ds = tf.data.Dataset.load(train_path)
test_ds = tf.data.Dataset.load(test_path)

'''--------------------------------------------------------------------'''

'''Preprocessing Pipeline'''
def preprocessing(data, batch_size):
  '''
        Preprocessing function that preprocesses the loaded datasets.

        Parameters:
            data: a dataset with image-target-tuples 
        
        Returns:
            ds: a mapped, shuffled, batched and prefetched dataset
  '''

  #convert data from uint8 to float32
  ds = data.map(lambda img, target: (tf.cast(img, tf.float32), (tf.cast(target, tf.float32))))
  #normalize image values
  ds = ds.map(lambda img, target: ((img/128)-1., target))
  #cache this progress in memory
  ds = ds.cache()
  #shuffle, prefetch, batch
  ds = ds.shuffle(1000)
  ds = ds.batch(32)
  ds = ds.prefetch(20)

  return ds

'''--------------------------------------------------------------------'''

'''The Model
   This shows that different initializer perform equally well: 
   https://wandb.ai/sauravm/Regularization-LSTM/reports/Neural-Network-Initialization-With-Keras--VmlldzoyMTI5NDYx
'''

# Basic CNN 12 conv layers using L1 Regularizer, 6 pooling layers and a dense layer 
class BasicConvRegL(tf.keras.Model):
    '''
        Class that builds a basic CNN with 6 conv layers using L1 regularization. Optimization technique: Adam
    '''

    def __init__(self, learning_rate):
        super(BasicConvRegL, self).__init__()
        tf.random.set_seed(0)

        # regularizer..
        reg = tf.keras.regularizers.L1(0.001)

        self.convlayer1 = tf.keras.layers.Conv2D(input_shape=(43, 43, 3), filters=43, kernel_size=3, padding='same', activation='relu', kernel_regularizer=reg)
        self.convlayer2 = tf.keras.layers.Conv2D(filters=43, kernel_size=3, padding='same', activation='relu', kernel_regularizer=reg)
        self.pooling1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

        self.convlayer3 = tf.keras.layers.Conv2D(filters=86, kernel_size=3, padding='same', activation='relu', kernel_regularizer=reg)
        self.convlayer4 = tf.keras.layers.Conv2D(filters=86, kernel_size=3, padding='same', activation='relu', kernel_regularizer=reg)
        self.pooling2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

        self.convlayer5 = tf.keras.layers.Conv2D(filters=172, kernel_size=3, padding='same', activation='relu', kernel_regularizer=reg)
        self.convlayer6 = tf.keras.layers.Conv2D(filters=172, kernel_size=3, padding='same', activation='relu', kernel_regularizer=reg)
        self.global_pool = tf.keras.layers.GlobalAvgPool2D()

        self.out = tf.keras.layers.Dense(91, activation='sigmoid')

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate,decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

        #set metrics
        self.metrics_list = [tf.keras.metrics.Precision(name="precision"),
                             tf.keras.metrics.Recall(name="recall"),
                             tf.keras.metrics.Mean(name='loss')]

        #set loss function
        self.loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # call function initializing the layers
    def call(self, x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.pooling1(x)
        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = self.pooling2(x)
        x = self.convlayer5(x)
        x = self.convlayer6(x)
        x = self.global_pool(x)
        x = self.out(x)
        return x

    

    @property
    # return a list with all metrics in the model to collect accuracies
    def metrics(self):
        return self.metrics_list

    # reset all metrics objects
    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    # train step method
    def train_step(self, data): 
        img, label = data
        
        with tf.GradientTape() as tape:
            output = self((img), training=True)
            loss = self.loss_function(label, output)
            
        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      
        
        # update the state of the metrics according to loss
        self.metrics[0].update_state(label, output)
        self.metrics[1].update_state(label, output)
        self.metrics[2].update_state(loss)


        # return a dictionary with metric names as keys and metric results as values
        return {m.name : m.result() for m in self.metrics}


    # test_step method
    def test_step(self, data):
        img, label = data

        # same as train step (without parameter updates)
        output = self((img), training=True)
        loss = self.loss_function(label, output)

        self.metrics[0].update_state(label, output)
        self.metrics[1].update_state(label, output)
        self.metrics[2].update_state(loss)

        return {m.name : m.result() for m in self.metrics}

def create_summary_writers(config_name):
    '''
        Function that creates a summary writer to keep an eye on the parameter development

        Parameters:
            config_name: string that holds the configuration name that is added to the path where the logs are saved
        
        Returns:
            train_summary_writer: a summary writer for the train dataset
            val:summary_writer: a summary writer for the validation dataset
    '''

    tf.random.set_seed(0)
    
    # define where to save the logs
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_path = f"/content/drive/MyDrive/model_data/logs/{config_name}/{current_time}/training"
    val_log_path = f"/content/drive/MyDrive/model_data/logs/{config_name}/{current_time}/test"

    # log writer for training metrics
    train_summary_writer = tf.summary.create_file_writer(train_log_path)

    # log writer for validation metrics
    val_summary_writer = tf.summary.create_file_writer(val_log_path)
    
    return train_summary_writer, val_summary_writer


'''--------------------------------------------------------------------'''

'''Training the Model'''
def training_loop(model, train_ds, test_ds, epochs, train_summary_writer, val_summary_writer):
  '''
        Training loop function that creates checkpoints and runs the training of the model for the givn epochs.

        Parameters:
            model: the model that should be trained and tested
            train_ds: dataset for training
            test_ds: dataset for testing
            epochs: Int that holds the number of epochs the model should be run
            train_summary_writer: a summary writer for the train dataset
            val:summary_writer: a summary writer for the validation dataset
            
        
        Returns:
            model: the model with the trained weights and parameters
            train_losses: array of the training losses for every epoch
            train_precision: array of the training precision for every epoch
            train_recall: array of the training recall for every epoch
            train_f1: array of the calculated f1 score of the training for every epoch
            test_losses: array of the test losses for every epoch
            test_precision: array of the test precision for every epoch
            test_recall: array of the test recall for every epoch
            test_f1: array of the calculated f1 score of the testing for every epoch
  '''

  # initialize parameters
  tf.random.set_seed(0)
  train_precision = []
  train_recall = []
  train_losses = []
  train_f1 = []
  test_precision = []
  test_recall = []
  test_losses = []
  test_f1 = []

  #initialize checkpoint and check whether there are saved checkpoints from where training is continued
  ckpt = tf.train.Checkpoint(step=tf.Variable(1))
  manager = tf.train.CheckpointManager(checkpoint=ckpt, directory='/content/drive/MyDrive/model_data/checkpoints', max_to_keep=3)

  # restore checkpoints or initialize from sratch
  ckpt.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")
  

  # iterate over epochs
  for e in range(epochs):
    print(f'Epoch: {str(e)}')

    # train steps on all batches in the traning dataset
    for data in train_ds:
      metrics = model.train_step(data)
      
      ckpt.step.assign_add(1)
      if int(ckpt.step) % 10 == 0:
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

    # log and print training metrics
    with train_summary_writer.as_default():
      # for scalar metrics:
      for metric in model.metrics:
        tf.summary.scalar(f"{metric.name}", metric.result(), step=e)

    
    # save the metrics items for the training
    for k, v in metrics.items():
      if "precision" in k:
        train_precision.append(round(v.numpy(),3))
        pres_v = float(v.numpy())
      elif "recall" in k:
        train_recall.append(round(v.numpy(),3))
        recall_v = float(v.numpy())
      elif "loss" in k:
        train_losses.append(round(v.numpy(),3))
    
    # calculate the f1 score for the training
    f1_train = 2*((pres_v*recall_v)/(pres_v+recall_v))
    train_f1.append(round(f1_train, 3))
    print(f"train_f1: %.2f" % round(f1_train, 3))
    
    print([f"train_{key}: {round(value.numpy(),3)}" for (key, value) in metrics.items()])
    print(train_f1)

    # reset metrics object
    model.reset_metrics()

    # evaluate on valdation data
    for test_data in test_ds:
      metrics_test = model.test_step(test_data)

    # log validation metrics
    with val_summary_writer.as_default():
      # for scalar metrics:
        for metric in model.metrics:
          tf.summary.scalar(f"{metric.name}", metric.result(), step=e)

    
    # save the metrics items for the testing
    for k, v in metrics_test.items():
      if "precision" in k:
        test_precision.append(round(v.numpy(),3))
        pres_v = float(v.numpy())
      elif "recall" in k:
        test_recall.append(round(v.numpy(),3))
        recall_v = float(v.numpy())
      elif "loss" in k:
        test_losses.append(round(v.numpy(),3))
    
    # calculate the f1 score for the training
    f1_test = 2*((pres_v*recall_v)/(pres_v+recall_v))
    test_f1.append(round(f1_test, 3))
    print(f"test_f1: %.2f" % round(f1_test, 3))
    print([f"test_{key}: {round(value.numpy(),3)}" for (key, value) in metrics_test.items()])
    print(test_f1)
    print("\n")

  return model, train_losses, train_precision, train_recall, train_f1, test_losses, test_precision, test_recall, test_f1

def training_function(train_ds, test_ds, epochs = 15, learning_rate = 0.001):
  '''
        Training function that initializes the summary writers, the model and runs the training loop.

        Parameters:
            train_ds: dataset for training
            test_ds: dataset for testing
            epochs: Int that holds the number of epochs the model should be run
            learning_rate: float that holds the learning rate of the model

        
        Returns:
            Starts the training loop and returns the returns of the training loop.
  '''

  # create the summary writers
  train_summary_writer, val_summary_writer = create_summary_writers(config_name = 'BasicConvRegL')

  # preprocess the train and test datasets
  prep_train_ds = preprocessing(train_ds)
  prep_test_ds = preprocessing(test_ds)

  # initialize the model
  model = BasicConvRegL(learning_rate)

  return training_loop(model, prep_train_ds, prep_test_ds, epochs, train_summary_writer, val_summary_writer)

'''--------------------------------------------------------------------'''

'''Run the Model'''

# Commented out IPython magic to ensure Python compatibility. --> tensorboard was running perfectily in colab
# remove all logs
# !rm -rf logs/

# start and save tensorboard logs to a defined path
# %tensorboard --logdir /content/drive/MyDrive/model_data/logs/BasicConvRegL

# call the training function ans save the returns
model, train_losses, train_precision, train_recall, train_f1, test_losses, test_precision, test_recall, test_f1 = training_function(train_ds=train_ds, test_ds=test_ds)

# save the model and it's trained parameters to a predefined path 
model.save('/content/drive/MyDrive/model_data/Model')