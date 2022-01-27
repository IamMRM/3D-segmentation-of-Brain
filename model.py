import os
import numpy as np
import nibabel as nib
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
from eval_helper import *
from metrics import *


class Unet(nn.Module):#Basic 3D U-net model
    def __init__(self, input_size, output_size, dropout_rate=0.2):
        super(Unet, self).__init__()
        # convolution1 down
        self.convolution1 = nn.Conv3d(in_channels=input_size,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        
        # normalization_dropout 1
        self.normalization1 = nn.InstanceNorm3d(32)
        self.dropout1 = nn.Dropout3d(p=dropout_rate)

        # max-pooling 1
        self.pooling1 = nn.Conv3d(in_channels=32,
                               out_channels=32,
                               kernel_size=2,
                               stride=2)
        # convolution2 down
        self.convolution2 = nn.Conv3d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        
        # normalization_dropout 2
        self.normalization2 = nn.InstanceNorm3d(64)
        self.dropout2 = nn.Dropout3d(p=dropout_rate)

        # max-pooling 2
        self.pooling2 = nn.Conv3d(in_channels=64,
                               out_channels=64,
                               kernel_size=2,
                               stride=2)
        # convolution3 down
        self.convolution3 = nn.Conv3d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        
        # normalization_dropout 3
        self.normalization3 = nn.InstanceNorm3d(128)
        self.dropout3 = nn.Dropout3d(p=dropout_rate)

        # max-pooling 3
        self.pooling3 = nn.Conv3d(in_channels=128,
                               out_channels=128,
                               kernel_size=2,
                               stride=2)
        # convolution4 down
        self.convolution4 = nn.Conv3d(in_channels=128,
                               out_channels=256,
                               kernel_size=3,
                               padding=1)
        
        # normalization_dropout 4
        self.normalization4 = nn.InstanceNorm3d(256)
        self.dropout4 = nn.Dropout3d(p=dropout_rate)

        # up-sample convolution4
        self.up1 = nn.ConvTranspose3d(in_channels=256,
                                      out_channels=128,
                                      kernel_size=2,
                                      stride=2)        
        # conv 5 (add up1 + convolution3)
        self.convolution5 = nn.Conv3d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        # up-sample convolution5
        self.up2 = nn.ConvTranspose3d(in_channels=128,
                                      out_channels=64,
                                      kernel_size=2,
                                      stride=2)
        # convolution6 (add up2 + convolution2) 
        self.convolution6 = nn.Conv3d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        # up 3
        self.up3 = nn.ConvTranspose3d(in_channels=64,
                                      out_channels=32,
                                      kernel_size=2,
                                      stride=2)
        # convolution7 (add up3 + convolution1)
        self.convolution7 = nn.Conv3d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        # convolution8 (classification)
        self.convolution8 = nn.Conv3d(in_channels=32,
                               out_channels=output_size,
                               kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = F.relu(self.convolution1(x))
        x1 = self.normalization1(x1)
        x1 = self.dropout1(x1)
        x1p = self.pooling1(x1)

        x2 = F.relu(self.convolution2(x1p))
        x2 = self.normalization2(x2)
        x2 = self.dropout2(x2)        
        x2p = self.pooling2(x2)

        x3 = F.relu(self.convolution3(x2p))
        x3 = self.normalization3(x3)
        x3 = self.dropout3(x3)
        x3p = self.pooling3(x3)
        
        x4 = F.relu(self.convolution4(x3p))
        x4 = self.normalization4(x4)
        x4 = self.dropout4(x4)

        # decoder
        up1 = self.up1(x4)
        x5 = F.relu(self.convolution5(up1 + x3))
        up2 = self.up2(x5)
        x6 = F.relu(self.convolution6(up2 + x2))
        up3 = self.up3(x6)
        x7 = F.relu(self.convolution7(up3 + x1))
        
        out = F.softmax(self.convolution8(x7), dim=1) 
        return out

"""from torchsummary import summary
model = Unet(input_size=1, output_size=4)
summary(model, input_size=(1,32,32,32))"""

"""x = torch.rand(input_train.shape)
print(x.shape)
model = Unet(input_size=1, output_size=4)
a = model(x)
print(a.shape)"""