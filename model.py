import torch, os, numba, typing, types
import os
import numpy as np
import torch.optim # for optimizers
import torch.nn as nn # for neural network modules
import torch.nn.functional as f # for non-linear function
from torch.utils.data import DataLoader, Dataset # for data handling

# Feedforward Neural Network Class
class net(nn.Module):

    prec: np.float1
    
    def __init__(
        self, 
        prec: type,  
        in_feat: int, 
        out_feat: int
    ) -> None:
        super().__init__()
        
        self.prec = prec
        self.torch_dtype = self._get_torch_dtype(prec)
        
        self.layers = nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.ReLU(),
            nn.Linear(out_feat, in_feat),
            nn.ReLU(),
            nn.Linear(in_feat, out_feat),
            nn.ReLU(),
            nn.Linear(out_feat, in_feat)
        )

    def fit(self, epoch_num: int):

        for i in range(epoch_num):
            
            forward_pass = self.layers.forward()
            print(forward_pass)
        
            backprop = torch.autograd.grad(forward_pass)
            print(backprop)

            print(f"Training loss: {loss}")
        

    #Pass it to hidden layer torch.nn.function where it does sigmoid activation etc 
