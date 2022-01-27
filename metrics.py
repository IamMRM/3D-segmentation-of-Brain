import os
import numpy as np
import nibabel as nib
import ants
import copy
import time
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from operator import add
from helper import *
from datagen import *
from model import *
from eval_helper import *

def true_pos(gt, mask):
    temp1 = np.array(gt).astype(dtype=np.bool)
    temp2 = np.array(mask).astype(dtype=np.bool)
    return np.count_nonzero(np.logical_and(temp1, temp2))#compute the number of true positive voxels between mask and gt
def DSC_seg(gt, mask):
    """
    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask

    Output:
    - (float) Voxelwise Dice coefficient between the input and gt mask
    """
    A = np.sum(np.array(gt).astype(dtype=np.bool))
    B = np.sum(np.array(mask).astype(dtype=np.bool))

    return 2.0 * true_pos(gt, mask) / (A + B) \
        if (A + B) > 0 else 0

from scipy.ndimage.morphology import binary_erosion as imerode
from sklearn.neighbors import NearestNeighbors

def eucl_distance(a, b):
    nbrs_a = NearestNeighbors(n_neighbors=1,algorithm='kd_tree').fit(a) if a.size > 0 else None
    nbrs_b = NearestNeighbors(n_neighbors=1,algorithm='kd_tree').fit(b) if b.size > 0 else None
    distances_a, _ = nbrs_a.kneighbors(b) if nbrs_a and b.size > 0 else ([np.inf], None)
    distances_b, _ = nbrs_b.kneighbors(a) if nbrs_b and a.size > 0 else ([np.inf], None)
    return [distances_a, distances_b]
def surface_distance(gt, mask, spacing=list((1, 1, 1))):#Compute the surface distance between the input mask and gt mask
    a = np.array(gt).astype(dtype=np.bool)
    b = np.array(mask).astype(dtype=np.bool)
    a_bound = np.stack(np.where(np.logical_and(a, np.logical_not(imerode(a)))), axis=1) * spacing
    b_bound = np.stack(np.where(np.logical_and(b, np.logical_not(imerode(b)))), axis=1) * spacing
    return eucl_distance(a_bound, b_bound)
def HD(gt, mask, spacing=(1, 1, 1)):
    distances = surface_distance(gt, mask, spacing)
    return np.max([np.max(distances[0]), np.max(distances[1])])

def PVE(gt, mask):
    A = np.sum(np.array(gt).astype(dtype=np.bool))
    B = np.sum(np.array(mask).astype(dtype=np.bool))
    pve = np.abs(float(B - A) / A)
    return pve