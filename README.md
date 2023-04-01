# Final_Project_IANNsTF_WS22-23

Welcome the GitGuh repository that holds all the code for the final project of the course 'Implementing Artificial Neural Networks with TensorFlow' (WS 22/23) by Anneke Büürma (995025) and Carmen Amme (994813).

We used the NSD Special100 dataset to compare different model architectures and their representations along early, middle and late layers by using Representational Similarity Analysis. We also made comparisons between model representations and fMRI brain data of the same dataset in early, middle and late areas of the visual hierarchy. We also implemented and trained 1 convolutional neural network ourselves to compare the emerging representations along the layers to the pre-trained models and fMRI brain data. 

In this repository you will find:

- The following py. files containing self-written functions for data analysis and model training:
  - preprocessing.py: All functions related to any preprocessing.
  - roi.py: File holds all functions regarding roi definition.
  - rsa.py: All functions related to Representational Similarity Analysis.
  - pretrained_models.py: Functions that deal with images and layer activations.
  - pretrained_resnet50_keras.py: File that initializes the pretrained ResNet50 from the Keras library.
  - pretrained_vgg16_keras.py: File that initializes the pretrained VGG16 from the Keras library.
  - self_trained_model_training.py: Script for our self-trained CNN.
  - ourmodel.py: Class to easily read in our model with weights to do further analysis.
  
- a pdf of the Final Project Report
- a pdf of the Meeting Summary
- a video of our project presentation

This is a link to the Google Drive folder that holds the fMRI data, the NSD special100 images, the training data for our model and all other necessary data for this project. https://drive.google.com/drive/folders/1q_SOJWm4U_diwAOS-K5ITq4StCRxFMOQ?usp=sharing. Further, please make sure that the shared IANNsTF-folder is located in you google drive account (click on the shared link and link it to "Meine Ablage" - "My Storage".

This is the link to the GoogleColab sheet that holds all the code to run the functions from the py. files. https://colab.research.google.com/drive/1lzurz4VYAnlwTW0IK9OfMJ1e4yvJ4pfB?usp=sharing 
Make sure to upload the py. files to GoogleColab before running the code. Also make sure to adjust the paths to the data. We encountered errors when just executing the code through this link - in this case, please just make a copy of the GoogleColab sheet and run it again.

This is a link to download the training data that was used to train our model. https://drive.google.com/file/d/1qw8MUwGxm8Yz899_SnnCSsMHVby-CYl-/view
