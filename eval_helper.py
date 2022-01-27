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
from metrics import *

def apply_padding(input_data, patch_size):
    patch_half = tuple([idx // 2 for idx in patch_size])
    padding = tuple((idx, size-idx) for idx, size in zip(patch_half,patch_size))
    padded_image = np.pad(input_data,padding,mode='constant',constant_values=0)
    return padded_image

def get_patches(input_data, centers, patch_size=(15, 15, 15)):
    patches = []
    list_of_tuples = all([isinstance(center, tuple) for center in centers])
    sizes_match = [len(center) == len(patch_size) for center in centers]
    if list_of_tuples and sizes_match:
        # apply padding to the input image and re-compute the voxel coordenates
        # according to the new dimension
        padded_image = apply_padding(input_data, patch_size)
        patch_half = tuple([idx // 2 for idx in patch_size])
        new_centers = [map(add, center, patch_half) for center in centers]
        # compute patch locations
        slices = [[slice(c_idx-p_idx, c_idx+s_idx-p_idx)
                   for (c_idx, p_idx, s_idx) in zip(center,
                                                    patch_half,
                                                    patch_size)]
                  for center in new_centers]

        # extact patches
        patches = [padded_image[tuple(idx)] for idx in slices]

    return np.array(patches)

def get_data_channels(image_path,scan_names,ref_voxels,patch_shape,normalize=False):
    scan_path = os.path.join(image_path, str(scan_names[0]))
    current_scan = nib.load(scan_path).get_data().squeeze()

    if normalize:
        current_scan = normalize_data(current_scan)

    patches = get_patches(current_scan,ref_voxels,patch_shape)    
    patches = np.expand_dims(patches, axis=1)

    return patches

def get_candidate_voxels(input_mask,  step_size):#Extract candidate patches.
    candidate_voxels = input_mask > 0#(256,128,256) its True False array

    voxel_coords = get_voxel_coordenates(input_mask,
                                          candidate_voxels,
                                          step_size=step_size) #saray scans me se aik ke 212 regions return kiye hein
    return voxel_coords

def get_inference_patches(scan_path, input_data, roi, patch_shape, step, normalize=True):
    # get candidate voxels
    mask_image = nib.load(os.path.join(scan_path, roi))

    ref_voxels = get_candidate_voxels(mask_image.get_data(),step)#(x,y,z)
    # input images stacked as channels
    test_patches = get_data_channels(scan_path, #ye main function hai. sari inference krne ka
                                     input_data,
                                     ref_voxels,
                                     patch_shape,
                                     normalize=normalize)


    return test_patches, ref_voxels

//***************************************************************************
def invert_padding(padded_image, patch_size):
    patch_half = tuple([idx // 2 for idx in patch_size])
    padding = tuple((idx, size-idx)
                    for idx, size in zip(patch_half, patch_size))

    return padded_image[padding[0][0]:-padding[0][1],
                        padding[1][0]:-padding[1][1],
                        padding[2][0]:-padding[2][1]]
                        
def reconstruct_image(input_data, centers, output_size):
    # apply a padding around edges before writing the results
    patch_size = input_data[0, :].shape
    out_image = apply_padding(np.zeros(output_size), patch_size)
    patch_half = tuple([idx // 2 for idx in patch_size])
    new_centers = [map(add, center, patch_half) for center in centers]
    # compute patch locations
    slices = [[slice(c_idx-p_idx, c_idx+s_idx-p_idx)
               for (c_idx, p_idx, s_idx) in zip(center,
                                                patch_half,
                                                patch_size)]
              for center in new_centers]

    # for each patch, sum it to the output patch and
    # then update the frequency matrix

    freq_count = np.zeros_like(out_image)

    for patch, slide in zip(input_data, slices):
        out_image[tuple(slide)] += patch
        freq_count[tuple(slide)] += np.ones(patch_size)

    # invert the padding applied for patch writing
    out_image = invert_padding(out_image, patch_size)
    freq_count = invert_padding(freq_count, patch_size)

    # the reconstructed image is the mean of all the patches
    out_image /= freq_count
    out_image[np.isnan(out_image)] = 0

    return out_image