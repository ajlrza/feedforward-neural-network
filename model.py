import torch, os, numba, typing, types
import os
import numpy as np
import torch.optim # for optimizers
import torch.nn as nn # for neural network modules
import torch.nn.functional as f # for non-linear function
from torch.utils.data import DataLoader, Dataset # for data handling

# Feedforward Neural Network Class
import torch
import torch.nn as nn
import numpy as np

class FNN(nn.Module):
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
        
        self.criterion = nn.BCEWithLogitsLoss()  
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def _get_torch_dtype(self, np_type):
        """Helper to map numpy types to PyTorch types."""
        mapping = {
            np.float16: torch.float16,
            np.float32: torch.float32,
            np.float64: torch.float64,
        }

        return mapping.get(np_type, torch.float32)

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor, epoch_num: int):

        inputs = inputs.to(self.torch_dtype)
        targets = targets.to(self.torch_dtype)
        
        self.train() 
        
        for epoch in range(epoch_num):
            self.optimizer.zero_grad()
            
            predictions = self.layers(inputs)
            
            loss = self.criterion(predictions, targets)
            loss.backward()
            
            print(f"Epoch {epoch+1}/{epoch_num} - Training loss: {loss.item():.4f}")
            self.optimizer.step()
            
        return self.layers

class cnn(nn.Module):

    prec: np.float

    def __init__(self, prec: np.float, in_feat, out_feat):
    
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
        
        self.criterion = nn.BCEWithLogitsLoss()  
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    
    @numba.jit
    def _process_img_pixels(self, pixels: list[float]):

        for row in pixels:
            for col in row:
                pass

        pass



    def _get_torch_dtype(self, np_type):
        """Helper to map numpy types to PyTorch types."""
        mapping = {
            np.float16: torch.float16,
            np.float32: torch.float32,
            np.float64: torch.float64,
        }

        return mapping.get(np_type, torch.float32)

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor, epoch_num: int):

        inputs = inputs.to(self.torch_dtype)
        targets = targets.to(self.torch_dtype)
        
        self.train() 
        
        for epoch in range(epoch_num):
            self.optimizer.zero_grad()
            
            predictions = self.layers(inputs)
            
            loss = self.criterion(predictions, targets)
            loss.backward()
            
            print(f"Epoch {epoch+1}/{epoch_num} - Training loss: {loss.item():.4f}")
            self.optimizer.step()
            
        return self.layers

class rnn(nn.Module):

    prec: np.float1
    
    def __init__(
        self, 
        prec: np.float16 | np.float32 | np.float64 | np.float96 | np.float128,
        in_feat: int, 
        out_feat: int) -> None:
        
        super().__init__()

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


    # Optimization Algorithm - SOON
        

    #Pass it to hidden layer torch.nn.function where it does sigmoid activation etc 

class gan(nn.Module):

    prec: np.float1
    
    def __init__(
        self, 
        prec: np.float16 | np.float32 | np.float64 | np.float96 | np.float128,
        in_feat: int, 
        out_feat: int) -> None:
        
        super().__init__()

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


    # Optimization Algorithm - SOON
        

    #Pass it to hidden layer torch.nn.function where it does sigmoid activation etc 

class lstm(nn.Module):

    prec: np.float1
    
    def __init__(
        self, 
        prec: np.float16 | np.float32 | np.float64 | np.float96 | np.float128,
        in_feat: int, 
        out_feat: int) -> None:
        
        super().__init__()

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


    # Optimization Algorithm - SOON
        

    #Pass it to hidden layer torch.nn.function where it does sigmoid activation etc 

class transf(nn.Module):

    prec: np.float1
    
    def __init__(
        self, 
        prec: np.float16 | np.float32 | np.float64 | np.float96 | np.float128,
        in_feat: int, 
        out_feat: int) -> None:
        
        super().__init__()

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


    # Optimization Algorithm - SOON
        

    #Pass it to hidden layer torch.nn.function where it does sigmoid activation etc 