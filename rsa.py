# -*- coding: utf-8 -*-
'''All functions related to Representational Similarity Analysis.

   Part of the Final Project for the Module "Implementing IANNs with TensorFlow" Winter Term 2022/2023
   Final Project by Carmen Amme (994813) and Anneke Büürma (995025)
'''

import numpy as np
import math
from math import sqrt
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def calc_dist(matrix, distance, descriptor=None):
  '''
  This function calculates a representational dissimilarity matrix based on a given distance measurement.

  Parameters:
    matrix: A numpy matrix containing the values to calculate the distances for.
    distance: string of the distance measurement ('euclidean', or 'pearson')

  Returns:
    a numpy representational dissimilarity matrix 
  '''

  if distance == 'euclidean':
    first_iter_i = True
    for i in range(matrix.shape[-1]):
      vec1 = matrix[:, i]
      
      first_iter_j = True
      for j in range(matrix.shape[-1]):
        vec2 = matrix[:, j]
        euc_dist = sqrt(sum((val1 - val2)**2 for val1, val2 in zip(vec1, vec2)))

        if first_iter_j:
          euc_arr = np.array(euc_dist)
          first_iter_j = False
        else:
          euc_arr = np.append(euc_arr, euc_dist)
    
      if first_iter_i:
        rdm = euc_arr
        first_iter_i = False
      else:
        rdm = np.vstack((rdm, euc_arr))
  
  if distance == 'pearson':
    rdm = 1-(np.corrcoef(matrix, rowvar=False))

  return rdm

def plot_three_rdms(early, middle, late, distance, description=None):
  '''
  This function plots three RDMs next to each other.

  Parameters:
    early: A numpy matrix containing the values to calculate the distances for.
    middle: A numpy matrix containing the values to calculate the distances for.
    late: A numpy matrix containing the values to calculate the distances for.
    distance: string of the distance measurement ('euclidean', or 'pearson')
    description: string of the description of the RDMs (e.g. model or fMRI data)
  '''

  fig, ax = plt.subplots(1, 3, sharey=True, figsize=(15, 5))

  img1 = ax[0].imshow(early, cmap='viridis', vmin=0, vmax=1)
  img2 = ax[1].imshow(middle, cmap='viridis',vmin=0, vmax=1)
  img3 = ax[2].imshow(late, cmap='viridis',vmin=0, vmax=1) 

  ax[0].set_title('early')
  ax[1].set_title('middle')
  ax[2].set_title('late')
  fig.suptitle('Representational Disimilaity Matrices - {} Correlation - {}'.format(distance, description), fontsize=15)

  plt.colorbar(img1, ax=ax[0], fraction=0.046, pad=0.04)
  plt.colorbar(img2, ax=ax[1], fraction=0.046, pad=0.04)
  plt.colorbar(img3, ax=ax[2], fraction=0.046, pad=0.04)
  plt.show()


def second_order_rdm(rdms):
  '''
  This function calculates a second order RDM based Spearman Rank correlation.

  Parameters:
    rdms: A list of numpy matrices containing the values to calculate the distances for.
  
  Returns:
    a numpy second order RDM
  '''

  flattened = [m.flatten() for m in rdms]

  flattened = np.array(flattened).transpose()

  # spearman rank correlation matrix
  spearman_corr = spearmanr(flattened)[0]

  # Compute RDM: 1 - correlation
  sec_rdm = np.ones(spearman_corr.shape) - spearman_corr

  return sec_rdm


def plot_sec_rdm(rdms, labels):
  '''
  This function plots a second order RDM.

  Parameters:
    rdms: A numpy matrix containing the values to calculate the distances for.
    labels: A list of labels as strings for the axes; 
            note that this has the be the same order and length as the list of rdms
  '''

  sec_rdms = second_order_rdm(rdms)

  fig, ax = plt.subplots(1,1)
  img1 = ax.imshow(sec_rdms, vmin=0, vmax=1)
  plt.colorbar(img1, ax=ax, fraction=0.046, pad=0.04)

  ticks_y = np.arange(len(labels))
  ticks_x = [i-1 for i in ticks_y]
  ax.set_xticks(ticks_x)
  plt.xticks(rotation=45)
  ax.set_xticklabels(labels)
  ax.set_yticks(ticks_y)
  ax.set_yticklabels(labels)
  fig.suptitle('Second Order RDM')