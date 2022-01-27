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
from helper import *
from model import *
from eval_helper import *
from metrics import *

def normalize_data(im):
    min_int = abs(im.min())
    max_int = im.max()
    if im.min() < 0:#again find out the min
        im = im.astype(dtype=np.float32) + min_int
        im = im / (max_int + min_int)
    else:
        im = (im.astype(dtype=np.float32) - min_int) / max_int
    return im

def get_voxel_coordenates(input_data,roi,random_pad=(0, 0, 0),step_size=(1, 1, 1)):
    # compute initial padding
    r_pad = np.random.randint(random_pad[0]+1) if random_pad[0] > 0 else 0
    c_pad = np.random.randint(random_pad[1]+1) if random_pad[1] > 0 else 0
    s_pad = np.random.randint(random_pad[2]+1) if random_pad[2] > 0 else 0

    # precompute the sampling points based on the input
    sampled_data = np.zeros_like(input_data)
    for r in range(r_pad, input_data.shape[0], step_size[0]):
        for c in range(c_pad, input_data.shape[1], step_size[1]):
            for s in range(s_pad, input_data.shape[2], step_size[2]):
                sampled_data[r, c, s] = 1

    # apply sampled points to roi and extract sample coordenates
    [x, y, z] = np.where(roi * sampled_data.squeeze())

    # return as a list of tuples
    return [(x_, y_, z_) for x_, y_, z_ in zip(x, y, z)]


class MRI_DataPatchLoader(Dataset):
  def __init__(self,input_data,labels,rois,patch_size,apply_padding,normalize,sampling_type,sampling_step,transform=None):
    self.patch_size = patch_size
    self.patch_half = tuple([idx // 2 for idx in patch_size])
    self.input_scans, self.label_scans, self.roi_scans = self.load_scans(input_data,labels,rois,apply_padding)
    #print(len(self.input_scans))#10
    #print(self.input_scans[0][0].shape)#(256, 128, 256, 1) or in padded (288, 160, 288)
    self.input_train_dim = (1, ) + self.patch_size#(1,32,32,32)
    self.input_label_dim = (1, ) + self.patch_size#(1,32,32,32)

    if normalize:
      for i in range(len(self.input_scans)):
        self.input_scans[i][0] = normalize_data(self.input_scans[i][0])

    self.sampling_type=sampling_type
    self.sampling_step=sampling_step
    # Build the patch indexes based on the image index and the voxel coordenates
    self.patch_indexes = self.generate_patch_indexes()
    print('> DATA: Training sample size:', len(self.patch_indexes))
    self.transform=transform

  def __len__(self):
    return len(self.patch_indexes)

  def __getitem__(self, idx):
    im_ = self.patch_indexes[idx][0]#kon se input_scan se uthai hui h iss idx per
    center = self.patch_indexes[idx][1]#pora tuple

    slice_ = [slice(c_idx-p_idx, c_idx+s_idx-p_idx)
              for (c_idx, p_idx, s_idx) in zip(center,self.patch_half,self.patch_size)]
    # get current patches for both training data and labels
    input_train = np.stack([self.input_scans[im_][0][tuple(slice_)]], axis=0)
    input_label = np.expand_dims(self.label_scans[im_][0][tuple(slice_)], axis=0)

    # check dimensions and put zeros if necessary
    if input_train.shape != self.input_train_dim:
        print('error in patch', input_train.shape, self.input_train_dim)
        input_train = np.zeros(self.input_train_dim).astype('float32')
    if input_label.shape != self.input_label_dim:
        print('error in label')
        input_label = np.zeros(self.input_label_dim).astype('float32')
    if self.transform:
        input_train, input_label = self.transform([input_train,input_label])
    return input_train, input_label

  def apply_padding(self,input_data):#Apply padding to edges in order to avoid overflow
    padding = tuple((idx, size-idx) for idx, size in zip(self.patch_half, self.patch_size))#((16, 16), (16, 16), (16, 16))
    padded_image = np.pad(input_data,padding,mode='constant',constant_values=0)#0 value as background
    return padded_image

  def load_scans(self,input_data,label_data,roi_data,apply_padding=True):
    """Applying padding to input scans. Loading simultaneously input data and
        labels in order to discard missing data in both sets."""
    input_scans=[]
    label_scans=[]
    roi_scans=[]

    for s in input_data.keys():
      if apply_padding:
        input_ = [self.apply_padding((nib.load(input_data[s][0]).get_data().astype('float32')).squeeze())]
        label_ = [self.apply_padding((nib.load(label_data[s][0]).get_data().astype('float32')).squeeze())]
        roi_ = [self.apply_padding((nib.load(roi_data[s][0]).get_data().astype('float32')).squeeze())]
        input_scans.append(input_)
        label_scans.append(label_)
        roi_scans.append(roi_)
      else:
        input_ = [(nib.load(input_data[s][0]).get_data().astype('float32')).squeeze()]
        label_ = [(nib.load(label_data[s][0]).get_data().astype('float32')).squeeze()]
        roi_ = [(nib.load(roi_data[s][0]).get_data().astype('float32')).squeeze()]
        input_scans.append(input_)
        label_scans.append(label_)
        roi_scans.append(roi_)
      print('> DATA: Loaded scan', s, 'roi size:',np.sum(roi_[0] > 0),' label_size: ', np.sum(label_[0] > 0))
    return input_scans, label_scans, roi_scans


  def generate_patch_indexes(self):
    training_indexes=[]
    for s,l,r,i in zip(self.input_scans,self.input_scans,self.roi_scans,range(len(self.input_scans))):
      candidate_voxels = self.get_candidate_voxels(s[0], l[0], r[0])
      voxel_coords = get_voxel_coordenates(s[0],candidate_voxels,step_size=self.sampling_step)
      training_indexes += [(i, tuple(v)) for v in voxel_coords]
    return training_indexes

  def get_candidate_voxels(self,input_mask, label_mask, roi_mask):
    if self.sampling_type == 'all':
      sampled_mask = input_mask > 0
    elif self.sampling_type == 'mask':
      sampled_mask = roi_mask > 0

    return sampled_mask