import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from complexPyTorch.complexFunctions import complex_relu

class ComplexActivation(nn.Module):
    def __init__(self, fn = complex_relu):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = self.fn(x)
        return x

class Conv1x1LogMatrixRegressor(nn.Module):
    """
    assumes an input of shape [batch_size, h, w, 2, 2]
    """
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
                 nn.Conv2d(in_channels=4, out_channels=20, 
                           kernel_size=1, stride=1, padding=0, 
                           dtype=torch.cfloat
                           ),
                 ComplexActivation(complex_relu),   


                 nn.Conv2d(in_channels=20, out_channels=50, 
                           kernel_size=1, stride=1, padding=0, 
                           dtype=torch.cfloat,
                           ),
                 ComplexActivation(complex_relu), 

                 nn.Conv2d(in_channels=50, out_channels=20, 
                           kernel_size=1, stride=1, padding=0, 
                           dtype=torch.cfloat,
                           ),
                 ComplexActivation(complex_relu), 

                 nn.Conv2d(in_channels=20, out_channels=9, 
                           kernel_size=1, stride=1, padding=0, 
                           dtype=torch.cfloat,
                           )
        )
        
    
    def forward(self, x):
        
        # expected input shape is [batch_size, h, w, 2,2]
        # flatten to [batch_size, h, w, 4]
        s = x.shape
        x = x.reshape(*s[:-2], np.product(s[-2:]))
        
        # permute dims to a shape of [batch_size, 4, h, w] (torch needs channels first)
        x = torch.permute(x, [0,3,1,2])

        #for layer in self.layers:
        #    x = layer(x)
        x = self.layers(x)


        # the output is [batch_size, h, w, 9] and we reshape it to [batch_size, h, w, 3, 3]
        x = torch.permute(x, [0,2,3,1])
        x = x.reshape(*x.shape[:-1], 3, 3)
        
        return x
        