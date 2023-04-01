# -*- coding: utf-8 -*-
'''All functions related to any preprocessing.
   
   Part of the Final Project for the Module "Implementing IANNs with TensorFlow" Winter Term 2022/2023
   Final Project by Carmen Amme (994813) and Anneke Büürma (995025)
'''

import pandas as pd
import numpy as np
import nibabel as nib
import os
import tensorflow as tf


def get_special100(special100_path, response_path):
  ''' This function creates a pandas dataframe that contains image IDs for all special100 images, including their position per session. 
      This information is needed to extract the corresponding trial-based fMRI data. 

      Parameters:
          sepcial100_path: path to the folder where table (.tsv) is stored that holds image IDs of special100 images, as downloaded from https://natural-scenes-dataset.s3.amazonaws.com/index.html#nsddata/stimuli/nsd/
          response_path: path to the folder where table (.tsv) is stored that holds trial numbers and presented images, as downloaded from https://natural-scenes-dataset.s3.amazonaws.com/index.html#nsddata/ppdata/subj01/behav/

      Returns:
          A dataframe that includes only special100-trials including a column that hold information about which trial in each session the image was shown.

  '''

  special100 = pd.read_table(special100_path, header=None)
  special100 = special100.squeeze().tolist()
  res = pd.read_table(response_path)

  #we create a column which counts the trials for each session
  first_it = True
  for i in np.unique(res['SESSION']):
    df_sess = res[res['SESSION'] == i].copy(deep=True)  
    df_sess['TRIAL_PER_SESS'] = range(len(df_sess))
    df_sess['TRIAL_PER_SESS'] += 1 #add 1, because we want the counter to start at 1
    if first_it:
      res_added = pd.DataFrame()
      first_it = False
    else:
      res_added = pd.concat([res_added, df_sess], ignore_index=True)
  
  #turn this into a pd series
  tr_per_sess = pd.DataFrame(res_added, columns = ['TRIAL_PER_SESS'])
  tr_per_sess = tr_per_sess.squeeze()

  #get the responses for all special100
  res_special100 = res[res['73KID'].isin(special100)]

  #we only take the trials where subjects saw the images for the first time
  res_special100 = res_special100[res_special100['ISOLD'] == 0]

  #insert the series with trial_per_session count based on index
  res_special100 = res_special100.join(tr_per_sess)

  res_special100 = res_special100.sort_values(['SESSION', 'TRIAL_PER_SESS'])

  res_special100 = res_special100.reset_index(drop=True)

  return res_special100



def save_special100_fMRI_session(res_special100, file_path, session, subject):
  '''This function extracts and saves fMRI data of trials that were part of the NSD special100 for each trial. 
        This way, the data take up less soace and it will be quicker to read them import them.
    
     Parameters:
        res_special100: A dataframe with responses of all special100 images, as returned from the function get_special_100.
        file_path: Path to the subject folder where fMRI data per trial is stored and saved.
        session: For which session the fMRI data are to be extracted.
        subject: For which subject the fMRI data is to be extracted.
        
     Returns:
        nothing

  '''

  session2d = "%02d" % (session,)

  try:
    mri_file = os.path.join(file_path, 'sub' + str(subject), f'betas_session{session2d}.nii.gz')
    img = nib.load(mri_file)
  except:
    mri_file = os.path.join(file_path, 'sub' + str(subject), f'betas_session{session2d}.nii')
    img = nib.load(mri_file)
  img_data = img.get_fdata()

  print('img data total session shape', img_data.shape)

  this_trial = res_special100.loc[(res_special100.SESSION == session), ['SUBJECT', 'SESSION', 'TRIAL_PER_SESS', '73KID']]

  slices_list = []
  for i in range(len(this_trial)):
    time = this_trial.at[this_trial.index[i], 'TRIAL_PER_SESS']
    one_slice = img_data[:, :, :, time]
    slices_list.append(one_slice)

  stacked_slices = np.stack(slices_list, axis=3)
  print('special100 data this trial', stacked_slices.shape)

  stacked_slices = nib.Nifti1Image(stacked_slices.astype(np.uint8), img.affine)
  nib.save(stacked_slices, os.path.join(file_path, 'sub' + str(subject), f'betas_special100_session{session2d}.nii.gz'))

  print('saved and done')


def order_special100(res_special100, as_string):
  ''' This function orders the special100 images in the order they were presented to subjects.
    
      Parameters:
          res_special100: A dataframe with responses of all special100 images, as returned from the function get_special_100.
          as_string:  If True, the image IDs are returned as strings consisting of 5 digits. 
                        If False, the image IDs are returned as integers.
        
      Returns:
          A list with the image IDs of the special100 images in the order they were presented to subjects.

  '''
  order = res_special100["73KID"].values.tolist()

  if as_string:
    counter = 0
    for i in order:
      i = str(i).zfill(5)
      order[counter] = i
      counter += 1

  return order


def import_special100_fMRI(file_path):
  ''' This function imports all fMRI data of special100 images. For the order of the images run function order_special100.
      
        Parameters:
            file_path: Path to the subject folder where fMRI data per session for special100 is stored and saved.
          
        Returns:
            A numpy array with all fMRI data of special100 images in the order they were presented to subjects.
  
  '''
  if not os.path.exists(os.path.join(file_path, 'betas_special100.nii.gz')):
    print('file does not exist, creating it now')
    first_iter = True
    for file in os.listdir(file_path):
      if 'betas_special100' in file:
        img = nib.load(os.path.join(file_path, file))
        img_data = img.get_fdata()
        if first_iter:
          fmri_special100 = img_data
          first_iter = False
        else:
          fmri_special100 = np.concatenate((fmri_special100, img_data), axis=-1)

      #save file
      fmri_special100 = nib.Nifti1Image(fmri_special100.astype(np.uint8), img.affine)
      nib.save(fmri_special100, os.path.join(file_path, 'betas_special100.nii.gz'))
  
  #if file exists, just load it
  else:
    print('file exists, loading it now')
    img = nib.load(os.path.join(file_path, 'betas_special100.nii.gz'))
    fmri_special100 = img.get_fdata()
  
  return fmri_special100


def load_trained_model(model_path):
  '''
      Load our self-trained model.
  '''
  model = tf.keras.models.load_model(model_path)
  model.compile()

  return model