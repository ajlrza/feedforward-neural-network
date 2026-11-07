import torch, os, numba, typing, types
import os
import numpy as np
import torch.optim as opt # for optimizers
import torch.nn as nn # for neural network modules
import torch.nn.functional as f # for non-linear function
from torch.utils.data import DataLoader, Dataset # for data handling

# Feedforward Neural Network Class
class net(nn.Module):
    
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

    def tokenize(params):
        pass
    #BRO SO ALL THIS TIME WONDERING HOW OPTIMIZER WOULD FIT IN THE BIGGER PICTURE
    # I REALIZED I WAS MISSING THE TENSORS ITSELF TO WORK ON
    # WHICH NEEDS TO BE TOKENIZED/ENCODED FIRST. SO IT WAS AN INDIRECT MISS OF BIGGER PICURE
    # AND THE ELEPHANT IN THE ROOM WAS TENSORS, SO BASICALLY THE TRAIN METHOD EXPECTED IT ON THE USER CALL...

    def train(self, x_train, y_train, epoch_num: int):      

        self.optim = opt.Optimizer(x_train)

        for i in range(epoch_num):
            
            forward = self.layers.forward(x_train)
            print(forward)
            loss = nn.BCELoss(forward)
            self.optim.zero_grad()
            backward = torch.autograd.backward(loss)
            grads = torch.autograd.grad(outputs=backward, inputs=forward)
            self.optim.step()
            print(f"Training loss: {loss}")
