import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from complexPyTorch.complexFunctions import complex_relu
from complexPyTorch.complexLayers import ComplexBatchNorm2d


class ComplexActivation(nn.Module):
    def __init__(self, fn = complex_relu):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = self.fn(x)
        return x

smap = {'Shh': [0,0], 'Shv': [0,1], 'Svh': [1,0], 'Svv': [1,1]}

class Scatter2Coherence(nn.Module):
    """
    assumes an input of shape [batch_size, h, w, 2, 2]
    """
    def __init__(self, in_channels):
        super().__init__()
        
        self.in_channels = in_channels

        self.layers = nn.Sequential(
                 nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, 
                           kernel_size=5, stride=5, padding=0, 
                           dtype=torch.cfloat
                           ),
                 #ComplexActivation(complex_relu),   
                 #ComplexBatchNorm2d(10),
        
                 nn.Conv2d(in_channels=self.in_channels, out_channels=10, 
                           kernel_size=5, stride=1, padding='same', 
                           dtype=torch.cfloat,
                           ),
                 ComplexActivation(complex_relu), 
                 ComplexBatchNorm2d(10),

                 nn.Conv2d(in_channels=10, out_channels=20, 
                           kernel_size=5
                           , stride=1, padding='same', 
                           dtype=torch.cfloat,
                           ),
                 ComplexActivation(complex_relu), 
                 ComplexBatchNorm2d(20),

                 nn.Conv2d(in_channels=20, out_channels=20, 
                           kernel_size=5
                           , stride=1, padding='same', 
                           dtype=torch.cfloat,
                           ),
                 ComplexActivation(complex_relu), 
                 ComplexBatchNorm2d(20),

                 nn.Conv2d(in_channels=20, out_channels=20, 
                           kernel_size=5
                           , stride=1, padding='same', 
                           dtype=torch.cfloat,
                           ),
                 ComplexActivation(complex_relu), 
                 ComplexBatchNorm2d(20),

                 nn.Conv2d(in_channels=20, out_channels=20, 
                           kernel_size=5
                           , stride=1, padding='same', 
                           dtype=torch.cfloat,
                           ),
                 ComplexActivation(complex_relu), 
                 ComplexBatchNorm2d(20),

                 nn.Conv2d(in_channels=20, out_channels=10, 
                           kernel_size=5, stride=1, padding='same', 
                           dtype=torch.cfloat,
                           ),
                 ComplexActivation(complex_relu), 
                 ComplexBatchNorm2d(10),

                 nn.Conv2d(in_channels=10, out_channels=5, 
                           kernel_size=5, stride=1, padding='same', 
                           dtype=torch.cfloat,
                           ),
                 ComplexActivation(complex_relu), 
                 ComplexBatchNorm2d(5),

                 nn.Conv2d(in_channels=5, out_channels=1, 
                           kernel_size=3, stride=1, padding='same', 
                           dtype=torch.cfloat,
                           ),
        )

    def get_output_shape(self, input_shape):
        x = torch.rand((1,self.in_channels, *input_shape)).type(torch.cfloat)
        return self(x).shape[-2:]
    
    def forward(self, x):
        
        x = self.layers(x)
        
        return x
        

class Scatter2Scatter(nn.Module):
    """
    assumes an input of shape [batch_size, h, w, 2, 2]
    """
    def __init__(self, in_channels):
        super().__init__()
        
        self.in_channels = in_channels

        self.layers = nn.Sequential(
                 nn.Conv2d(in_channels=self.in_channels, out_channels=20, 
                           kernel_size=1, stride=1, padding=0, 
                           dtype=torch.cfloat
                           ),
                 ComplexActivation(complex_relu),   
                 ComplexBatchNorm2d(20),
        
                 nn.Conv2d(in_channels=20, out_channels=20, 
                           kernel_size=5, stride=2, padding=0, 
                           dtype=torch.cfloat,
                           ),
                 ComplexActivation(complex_relu), 
                 ComplexBatchNorm2d(20),

                 nn.Conv2d(in_channels=20, out_channels=10, 
                           kernel_size=1, stride=1, padding=0, 
                           dtype=torch.cfloat,
                           ),
                 ComplexActivation(complex_relu), 
                 ComplexBatchNorm2d(10),


                 nn.Conv2d(in_channels=10, out_channels=10, 
                           kernel_size=5, stride=2, padding=0, 
                           dtype=torch.cfloat,
                           ),
                 ComplexActivation(complex_relu), 

                 nn.ConvTranspose2d(10, 10, kernel_size=10, stride=1, dtype=torch.cfloat),
                 ComplexActivation(complex_relu), 

                 nn.ConvTranspose2d(10, 5, kernel_size=8, stride=1, dtype=torch.cfloat),

                 ComplexActivation(complex_relu), 
                 nn.ConvTranspose2d(5, 1, kernel_size=6, stride=2, dtype=torch.cfloat)



        )
        
    def get_output_shape(self, input_shape):
        x = torch.rand((1,self.in_channels, *input_shape)).type(torch.cfloat)
        return self(x).shape[-2:]
    
    def forward(self, x):
        
        x = self.layers(x)
        
        return x