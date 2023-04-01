'''File holds all functions regarding roi definition.
   
   Part of the Final Project for the Module "Implementing IANNs with TensorFlow" Winter Term 2022/2023
   Final Project by Carmen Amme (994813) and Anneke Büürma (995025)
'''

# imports
import numpy as np
import nibabel as nib


def extract_rois(fmri_file_path, hcp_file_path, rois):
  ''' This function extracts cortical ROIs for 3 different levels in the visual hierarchy (Kietzmann et al., 2019) 
      based on parcellations of 180 cortical regions by Glasser et al. (2019). It keeps the value in the ROI and sets 
      all other values to 0. 
        early: V1, V2, V3
        middle: LO1-->20, LO2-->21, V4t-->156, LO3-->159
        late: IT/PHC (FFC-->18, PHA1-->126, PHA2-->127, TE1p-->133, TE2p-->136, VMV3-->154, PHA2-->155, VMV2-->160, VVC-->163)
  
  Parameters:
    img_data: 4D fMRI image data 
    fmri_file_path: path to the fmri nii-file
    hcp_file_path: path to the subject-specific .nii-file that holds voxel-wise parcellation for 180 cortical regions based on Glasser et al. (2019)
    rois: list of visual rois to extract --> possible are 'early', 'middle' and/or 'late'

  Returns:
    A dictonary of voxel-wise fMRI activation, one matrix per ROI (this makes it more flexible to return different numbers of ROIs)
  '''
  #load in fMRI data
  img = nib.load(fmri_file_path)
  img_data = img.get_fdata()
  img_data = img_data[:, :, :, :100]
  print('fMRI data has shape {}'.format(img_data.shape))


  #load in parcellation
  hcp_img = nib.load(hcp_file_path)
  hcp_data = hcp_img.get_fdata()
  print('parcellated anatomical image has shape {}'.format(hcp_data.shape))

  roi_dict = {}
  for roi in rois:
    if str(roi) == 'early':
      early_mask = np.where((hcp_data == 1) | (hcp_data == 4) | (hcp_data == 5), 1, np.nan)
      print('early', np.unique(early_mask, return_counts=True))

      early = np.zeros_like(img_data)
      second_dim = img_data.shape[-1]
      for i in range(second_dim):
        early[:, :, :, i] = img_data[:, :, :, i] * early_mask
      first_dim = np.count_nonzero(~np.isnan(early[:, :, :, 0]))
      early = early[~np.isnan(early)] #remove all nan values so that we only have voxels in ROI
      early = np.reshape(early, (first_dim, second_dim))
      roi_dict['early'] = early
    
    elif str(roi) == 'middle':
      middle_mask = np.where((hcp_data == 20) | (hcp_data == 21) | (hcp_data == 156) | (hcp_data == 159), 1, np.nan)
      print('middle', np.unique(middle_mask, return_counts=True))

      middle = np.zeros_like(img_data)
      second_dim = img_data.shape[-1]
      for i in range(second_dim):
        middle[:, :, :, i] = img_data[:, :, :, i] * middle_mask
      first_dim = np.count_nonzero(~np.isnan(middle[:, :, :, 0]))
      middle = middle[~np.isnan(middle)] #remove all nan values so that we only have voxels in ROI
      middle = np.reshape(middle, (first_dim, second_dim))
      roi_dict['middle'] = middle
  
    elif str(roi) == 'late':
      late_mask = np.where((hcp_data == 18) | (hcp_data == 126) | (hcp_data == 127) | 
                  (hcp_data == 133) | (hcp_data == 154) | (hcp_data == 155) | 
                  (hcp_data == 160) | (hcp_data == 163), 1, np.nan)
      print('late', np.unique(late_mask, return_counts=True))
      
      late = np.zeros_like(img_data)
      second_dim = img_data.shape[-1]
      for i in range(second_dim):
        late[:, :, :, i] = img_data[:, :, :, i] * late_mask
      first_dim = np.count_nonzero(~np.isnan(late[:, :, :, 0]))
      late = late[~np.isnan(late)] #remove all nan values so that we only have voxels in ROI
      late = np.reshape(late, (first_dim, second_dim))
      roi_dict['late'] = late

  return roi_dict

def count_voxels(rois):
  '''
  Parameters:
    rois: a list of ROIs for which to count voxels
  
  Prints voxel counts for each ROI'''

  for i in rois:
    print(rois[i], 'voxel count:', i.shape[0])