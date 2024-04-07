#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:28:55 2024

@author: alberto
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Callable, Optional


class ImageMatrixDatset(Dataset):
    def __init__(self, data_in: np.array,
                 data_out: np.array,
                 split_mask: np.array,
                 split: str = "train",
                 transform_in: Optional[Callable] = None,
                 transform_out: Optional[Callable] = None):
        if not (data_in.shape[:-2] == data_out.shape[:-2]):
            raise ValueError("data_in and data_out must have the same shape, except for the matrix dimension (last 2 dims)")
        if not (split_mask.shape == data_in.shape[:-2]):
            raise ValueError("split_mask must have the same shape as the data")
        
        self.transform_in = transform_in
        self.transform_out = transform_out
        self.img_shape = data_in.shape[:-2]
        self.matrix_in_shape = data_in.shape[-2:]
        self.matrix_out_shape = data_out.shape[-2:]
        self.splitmap = {'train': 0, 'test': 1, 'val': 2}
        self.split = split.lower()
        if self.split == "all":
            self.data_in = data_in.reshape((-1,) + self.matrix_in_shape)
            self.data_out = data_out.reshape((-1,) + self.matrix_out_shape)
        else:
            self.data_in = data_in[split_mask == self.splitmap[split]]
            self.data_out = data_out[split_mask == self.splitmap[split]]
        
    def __len__(self) -> int:
        return self.data_in.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_in = self.data_in[idx]
        sample_out = self.data_out[idx]
        if self.transform_in:
            sample_in = self.transform_in(sample_in)
        if self.transform_out:
            sample_out = self.transform_out(sample_out)
            
        return sample_in, sample_out


class GetMatrixElements(object):
    def __init__(self, pos_idx1, pos_idx2):
        if not (len(pos_idx1) == len(pos_idx2)):
            raise ValueError("pos_idx1 and pos_idx2 index lists must have the same size")

        self.pos_idx1 = pos_idx1
        self.pos_idx2 = pos_idx2
        
    def __call__(self, sample):
        return sample[..., self.pos_idx1, self.pos_idx2]

class GetMatrixElements_RealAndImag(object):
    def __init__(self, pos_idx1, pos_idx2):
        if not (len(pos_idx1) == len(pos_idx2)):
            raise ValueError("pos_idx1 and pos_idx2 index lists must have the same size")

        self.pos_idx1 = pos_idx1
        self.pos_idx2 = pos_idx2
        # Get all the elements i != j to include also imag part at the end
        poss = np.stack((pos_idx1, pos_idx2))
        self.posd = poss[:,poss[0] != poss[1]]
        
    def __call__(self, sample):
        return np.concatenate((sample.real[..., self.pos_idx1, self.pos_idx2],
                               sample.imag[..., self.posd[0], self.posd[1]]),
                              axis=-1)

class GetMatrixElements_RealAndImagTorch(object):
    def __init__(self, pos_idx1, pos_idx2):
        if not (len(pos_idx1) == len(pos_idx2)):
            raise ValueError("pos_idx1 and pos_idx2 index lists must have the same size")

        self.pos_idx1 = pos_idx1
        self.pos_idx2 = pos_idx2
        # Get all the elements i != j to include also imag part at the end
        poss = torch.stack((pos_idx1, pos_idx2))
        self.posd = poss[:,poss[0] != poss[1]]
        
    def __call__(self, sample):
        return torch.concatenate((sample.real[..., self.pos_idx1, self.pos_idx2],
                               sample.imag[..., self.posd[0], self.posd[1]]),
                              axis=-1)

class RecoverMatrix_From_RealAndImagElements(object):
    def __init__(self, matrix_size, pos_idx1, pos_idx2):
        if not (len(pos_idx1) == len(pos_idx2)):
            raise ValueError("pos_idx1 and pos_idx2 index lists must have the same size")

        self.pos_idx1 = pos_idx1
        self.pos_idx2 = pos_idx2
        self.matrix_size = matrix_size
        # Get all the elements i != j to index also imag part at the end
        poss = torch.stack((pos_idx1, pos_idx2))
        self.posd = poss[:,poss[0] != poss[1]]
        
    def __call__(self, sample):
        C = torch.zeros(sample.shape[:-1] + (self.matrix_size, self.matrix_size),
                        dtype=torch.complex64, device=sample.device)
        Nelems = len(self.pos_idx1)
        C.real[..., self.pos_idx1, self.pos_idx2] = sample[..., :Nelems]
        C.imag[..., self.posd[0], self.posd[1]] = sample[..., Nelems:]
        # Fill lower diagonal as hermitian
        for i in range(self.matrix_size):
            for j in range(i+1, self.matrix_size):
                C[..., j, i] = C[..., i,j].conj()
        return C

class SymmetricRevisedWishartLoss(object):
    def __init__(self, eps: float = 1e-5) -> None:
        if not (eps >= 0):
            raise ValueError("eps should be greater or equal to 0")
        self.eps = eps
    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Regularization of cov matrices with the given eps
        ri = self.eps * torch.eye(A.shape[-1], device=A.device)
        Ar = A + ri
        Br = B + ri
        # First trace
        d1 = torch.sum(torch.diagonal(torch.linalg.solve(Ar, Br).real, dim1=-2, dim2=-1), dim=-1).mean()
        # Second trace
        d2 = torch.sum(torch.diagonal(torch.linalg.solve(Br, Ar).real, dim1=-2, dim2=-1), dim=-1).mean()
        # Final result
        return (d1 + d2) / 2 - A.shape[-1]
    
class FrobeniusNormMeanSquaredLoss(object):
    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return (torch.linalg.norm(A-B, ord='fro', dim=(-2,-1))**2).mean()
