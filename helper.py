import os
import numpy as np
import nibabel as nib
import ants
import copy
import pandas as pd
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
from datagen import *
from model import *
from eval_helper import *
from metrics import *

def set_device():
  if torch.cuda.is_available():
    dev="cuda:0"
  else:
    dev="cpu"
  return torch.device(dev)

def mask_image(im):
  return (im > 0).astype('float32')

def imshow(inp, title=None):
    """Imshow for datagenerator patches"""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def dice_loss(input, target):
  smooth = 1.
  total_loss =0
  n_classes = 4
  for c in range(n_classes):
    tflat = torch.flatten(target ==c)
    iflat = torch.flatten(input[:,c,:,:,:])
    intersection = torch.sum(iflat*tflat)
    #a = (2. * intersection + smooth)
    loss = ( ((2. * intersection + smooth) /(torch.sum(iflat) + torch.sum(tflat) + smooth)))
    if c ==0:
      total_loss = loss
    else:
      total_loss = total_loss + loss
  total_loss = total_loss / n_classes

  return 1-total_loss

def calc_loss(pred, y,thresh=0.65):
  dice = dice_loss(pred, y)
  cross_entropy = F.cross_entropy(torch.log(torch.clamp(pred, 1E-7, 1.0)),y.squeeze(dim=1).long())

  loss = (thresh*dice) + ((1.0-thresh)*cross_entropy)
  return loss

def evaluation(training_path):
  test_scans = os.listdir(training_path)
  metrics = np.zeros((len(test_scans), 9))
  scan_id_list = []
  normalize=True
  th=0.5
  normalize=True
  for i, scan_name in enumerate(test_scans):
    real_path=os.path.join(training_path,scan_name)
    Image = ants.image_read(os.path.join(real_path, '{}.nii.gz'.format(scan_name)))
    infer_patches, coordenates = get_inference_patches(scan_path=real_path,input_data=['{}.nii.gz'.format(scan_name)],roi='{}_brainmask.nii.gz'.format(scan_name),patch_shape=patch_size,step=sampling_step,normalize=normalize)
    
    lesion_out = np.zeros((infer_patches.shape[0], 4, infer_patches.shape[2], infer_patches.shape[3],infer_patches.shape[4]), dtype='float32')#(212, 4, 32, 32, 32)
    batch_size =batchsize
    model.eval()
    with torch.no_grad():
      for b in range(0, len(lesion_out), batch_size):
        x = torch.tensor(infer_patches[b:b + batch_size]).to(device)
        pred = model(x)
        lesion_out[b:b + batch_size] = pred.cpu().numpy()
    
    # reconstruct image takes the inferred patches, the patches coordenates and the image size as inputs
    lesion_prob = np.expand_dims(reconstruct_image(lesion_out[:, 0],coordenates,Image.shape), axis=0)#(1, 256, 128, 256)
    lesion_prob2 = np.expand_dims(reconstruct_image(lesion_out[:, 1],coordenates,Image.shape), axis=0)
    lesion_prob3 = np.expand_dims(reconstruct_image(lesion_out[:, 2],coordenates,Image.shape), axis=0)
    lesion_prob4 = np.expand_dims(reconstruct_image(lesion_out[:, 3],coordenates,Image.shape), axis=0)
    tissue_seg = np.stack((lesion_prob, lesion_prob2, lesion_prob3, lesion_prob4), axis=0).squeeze()

    tissue_seg = np.argmax(tissue_seg, axis=0)

    CSF = tissue_seg == 1
    GM = tissue_seg == 2
    WM = tissue_seg == 3

    # binarize the results
    lesion_prob = (lesion_prob > th).astype('uint8')

    # evaluate the results
    gt = ants.image_read(os.path.join(real_path, '{}_seg.nii.gz'.format(scan_name)))
    dsc_CSF = DSC_seg(gt.numpy() == 1, CSF)
    dsc_GM = DSC_seg(gt.numpy() == 2, GM)
    dsc_WM = DSC_seg(gt.numpy() == 3, WM)

    hd_CSF = HD(gt.numpy() == 1, CSF)
    hd_GM = HD(gt.numpy() == 2, GM)
    hd_WM = HD(gt.numpy() == 3, WM)

    vd_CSF = PVE(gt.numpy() == 1, CSF)
    vd_GM = PVE(gt.numpy() == 2, GM)
    vd_WM = PVE(gt.numpy() == 3, WM)

    metrics[i] = [dsc_CSF, dsc_GM, dsc_WM, hd_CSF,hd_GM,hd_WM, vd_CSF,vd_GM,vd_WM]
    scan_id_list.append(scan_name)

    print('SCAN:', scan_name, 'dice_CSF: ', dsc_CSF, 'dice_GM:', dsc_GM, 'dice_WM:', dsc_WM, 'hd_CSF: ',hd_CSF,'hd_GM: ',hd_GM,'hd_WM: ',hd_WM,'vd_CSF: ',vd_CSF,'vd_GM: ',vd_GM,'vd_WM: ',vd_WM)

    # # save as nifti image is necessary
    seg_img = ants.from_numpy(tissue_seg.astype('uint8'))
    seg_img = ants.copy_image_info(Image, seg_img)
    # write segmented image to folder
    seg_path = "{}/{}.nii.gz".format(tmpdir, scan_name)
    #print(seg_path)
    ants.image_write(seg_img, seg_path)

  metrics_df = {'scan_id':scan_id_list, 'DSC_CSF':metrics[:, 0],'DSC_GM':metrics[:, 1],'DSC_WM':metrics[:, 2], 'hd_CSF':metrics[:, 3],'hd_GM':metrics[:, 4],'hd_WM':metrics[:, 5],
                'vd_CSF':metrics[:, 6],'vd_GM':metrics[:, 7],'vd_WM':metrics[:, 8]}
  m = pd.DataFrame(metrics_df, columns=['scan_id', 'DSC_CSF', 'DSC_GM', 'DSC_WM','hd_CSF','hd_GM','hd_WM','vd_CSF','vd_GM','vd_WM'])
  m_mean = m.describe().T
  return m_mean


def evaluation_test(test_path):
  test_scans = os.listdir(test_path)
  metrics = np.zeros((len(test_scans), 9))
  scan_id_list = []
  normalize=True
  th=0.5
  for i, scan_name in enumerate(test_scans):
    real_path=os.path.join(test_path,scan_name)
    Image = ants.image_read(os.path.join(real_path, '{}.nii.gz'.format(scan_name)))
    infer_patches, coordenates = get_inference_patches(scan_path=real_path,input_data=['{}.nii.gz'.format(scan_name)],roi='{}_brainmask.nii.gz'.format(scan_name),patch_shape=patch_size,step=sampling_step,normalize=normalize)
    
    lesion_out = np.zeros((infer_patches.shape[0], 4, infer_patches.shape[2], infer_patches.shape[3],infer_patches.shape[4]), dtype='float32')#(212, 4, 32, 32, 32)
    batch_size =batchsize
    model.eval()
    with torch.no_grad():
      for b in range(0, len(lesion_out), batch_size):
        x = torch.tensor(infer_patches[b:b + batch_size]).to(device)
        pred = model(x)
        lesion_out[b:b + batch_size] = pred.cpu().numpy()
    
    # reconstruct image takes the inferred patches, the patches coordenates and the image size as inputs
    lesion_prob = np.expand_dims(reconstruct_image(lesion_out[:, 0],coordenates,Image.shape), axis=0)#(1, 256, 128, 256)
    lesion_prob2 = np.expand_dims(reconstruct_image(lesion_out[:, 1],coordenates,Image.shape), axis=0)
    lesion_prob3 = np.expand_dims(reconstruct_image(lesion_out[:, 2],coordenates,Image.shape), axis=0)
    lesion_prob4 = np.expand_dims(reconstruct_image(lesion_out[:, 3],coordenates,Image.shape), axis=0)
    tissue_seg = np.stack((lesion_prob, lesion_prob2, lesion_prob3, lesion_prob4), axis=0).squeeze()

    tissue_seg = np.argmax(tissue_seg, axis=0)

    CSF = tissue_seg == 1
    GM = tissue_seg == 2
    WM = tissue_seg == 3

    # binarize the results
    lesion_prob = (lesion_prob > th).astype('uint8')
    # # save as nifti image is necessary
    seg_img = ants.from_numpy(tissue_seg.astype('uint8'))
    seg_img = ants.copy_image_info(Image, seg_img)
    # write segmented image to folder
    seg_path = "{}/{}.nii.gz".format(tmpdir, scan_name)
    #print(seg_path)
    ants.image_write(seg_img, seg_path)