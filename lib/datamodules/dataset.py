#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:28:55 2024

@author: alberto
"""
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from torch.types import _dtype
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

##############################################################################
# Parametrization based on taking some matrix elements
##############################################################################
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
    def __init__(self, pos_idx1, pos_idx2) -> None:
        if not (len(pos_idx1) == len(pos_idx2)):
            raise ValueError("pos_idx1 and pos_idx2 index lists must have the same size")

        self.pos_idx1 = pos_idx1
        self.pos_idx2 = pos_idx2
        # Get all the elements i != j to include also imag part at the end
        poss = torch.stack((pos_idx1, pos_idx2))
        self.posd = poss[:,poss[0] != poss[1]]
        
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
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
    
##############################################################################
# Parametrization based on Matrix trace normalization & Cholesky
##############################################################################
class Matrix_NormRhos_parametrization(object):
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        mdiag = torch.einsum("...ii->...i", sample).real
        di = mdiag / torch.sum(mdiag, dim=-1, keepdim=True)
        mdiag = 1.0 / torch.sqrt(mdiag)
        N = torch.einsum("...i,...j->...ij", mdiag, mdiag) * sample
        x1, x2 = torch.triu_indices(N.shape[-2], N.shape[-1], offset=1)
        rhos = N[..., x1, x2]
        return torch.cat((di, rhos.real, rhos.imag), dim=-1)
    
class Matrix_TraceNormRhos_parametrization(object):
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        mdiag = torch.einsum("...ii->...i", sample).real
        tr = torch.sum(mdiag, dim=-1, keepdim=True)
        di = mdiag / tr
        mdiag = 1.0 / torch.sqrt(mdiag)
        N = torch.einsum("...i,...j->...ij", mdiag, mdiag) * sample
        x1, x2 = torch.triu_indices(N.shape[-2], N.shape[-1], offset=1)
        rhos = N[..., x1, x2]
        return torch.cat((torch.log(tr), di, rhos.real, rhos.imag), dim=-1)
    
class Matrix_Cholesky_parametrization(object):
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        L = torch.linalg.cholesky(sample)
        ldiag = torch.einsum("...ii->...i", L).real
        x1, x2 = torch.tril_indices(L.shape[-2], L.shape[-1], offset=-1)
        offd = L[..., x1, x2]
        return torch.cat((ldiag, offd.real, offd.imag), dim=-1)
    
class RecoverMatrix_From_TraceNormRhos_parametrization(object):
    def __init__(self, matrix_size: int,
                 dtype: _dtype = torch.complex64) -> None:
        self.matrix_size = matrix_size
        self.dtype = dtype
        self.x1, self.x2 = torch.triu_indices(matrix_size, matrix_size,
                                              offset=1)
        
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        out_shape = sample.shape[:-1] + (self.matrix_size, self.matrix_size)
        C = torch.ones(out_shape, dtype=self.dtype, device=sample.device)
        # Recover params from sample
        tr = torch.exp(sample[..., 0])
        di = sample[..., 1:self.matrix_size+1]
        rhos = sample[..., self.matrix_size+1:]
        rhos = rhos[..., :len(self.x1)] + 1j*rhos[..., len(self.x1):]
        # Set outer diagonal elements to rhos
        C[..., self.x1, self.x2] = rhos
        # Fill lower diagonal as hermitian
        C[..., self.x2, self.x1] = rhos.conj()
        sdiag = torch.sqrt(di)
        N = torch.einsum("...i,...j->...ij", sdiag, sdiag)
        return tr[..., None, None] * N * C
    
class RecoverNormMatrix_From_NormRhos_parametrization(object):
    def __init__(self, matrix_size: int,
                 dtype: _dtype = torch.complex64) -> None:
        self.matrix_size = matrix_size
        self.dtype = dtype
        self.x1, self.x2 = torch.triu_indices(matrix_size, matrix_size,
                                              offset=1)
        
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        out_shape = sample.shape[:-1] + (self.matrix_size, self.matrix_size)
        C = torch.ones(out_shape, dtype=self.dtype, device=sample.device)
        # Recover params from sample
        di = sample[..., :self.matrix_size]
        rhos = sample[..., self.matrix_size:]
        rhos = rhos[..., :len(self.x1)] + 1j*rhos[..., len(self.x1):]
        # Set outer diagonal elements to rhos
        C[..., self.x1, self.x2] = rhos
        # Fill lower diagonal as hermitian
        C[..., self.x2, self.x1] = rhos.conj()
        sdiag = torch.sqrt(di)
        N = torch.einsum("...i,...j->...ij", sdiag, sdiag)
        return N * C
    
class TraceNormRhosActivarion(nn.Module):
    def __init__(
            self,
            matrix_size: int,
            dtype: _dtype = torch.complex64,
            **kwargs,
            ) -> None:
        super(TraceNormRhosActivarion, self).__init__()
        self.matrix_size = matrix_size
        self.dtype = dtype
        self.x1, self.x2 = torch.triu_indices(matrix_size, matrix_size,
                                              offset=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.moveaxis(x, -3, -1)
        # Recover params from x
        tr = x[..., 0:1]
        di = x[..., 1:self.matrix_size+1]
        rhos = x[..., self.matrix_size+1:]
        rhos = rhos[..., :len(self.x1)] + 1j*rhos[..., len(self.x1):]
        # make tr always positive
        tr = torch.exp(tr)
        # make di between [0, 1] and sum = 1
        di = torch.nn.functional.softmax(di, dim=-1)
        # make rhos abs() between [0,1] (softplus)
        rhos = rhos / (1+torch.abs(rhos))
        return torch.moveaxis(torch.cat((tr, di, rhos.real, rhos.imag), dim=-1), -1, -3)
    
class NormRhosActivarion(nn.Module):
    def __init__(
            self,
            matrix_size: int,
            dtype: _dtype = torch.complex64,
            **kwargs,
            ) -> None:
        super().__init__()
        self.matrix_size = matrix_size
        self.dtype = dtype
        self.x1, self.x2 = torch.triu_indices(matrix_size, matrix_size,
                                              offset=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.moveaxis(x, -3, -1)
        # Recover params from x
        di = x[..., :self.matrix_size]
        rhos = x[..., self.matrix_size:]
        rhos = rhos[..., :len(self.x1)] + 1j*rhos[..., len(self.x1):]
        # make di between [0, 1] and sum = 1
        di = torch.nn.functional.softmax(di, dim=-1)
        # make rhos abs() between [0,1] (softplus)
        rhos = rhos / (1+torch.abs(rhos))
        return torch.moveaxis(torch.cat((di, rhos.real, rhos.imag), dim=-1), -1, -3)
    
class RecoverMatrix_From_Cholesky_parametrization(object):
    def __init__(self, matrix_size: int,
                 dtype: _dtype = torch.complex64) -> None:
        self.matrix_size = matrix_size
        self.dtype = dtype
        self.x1, self.x2 = torch.tril_indices(matrix_size, matrix_size,
                                              offset=-1)
        
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        out_shape = sample.shape[:-1] + (self.matrix_size, self.matrix_size)
        L = torch.zeros(out_shape, dtype=self.dtype, device=sample.device)
        # Recover params from sample
        di = sample[..., :self.matrix_size]
        offd = sample[..., self.matrix_size:]
        offd = offd[..., :len(self.x1)] + 1j*offd[..., len(self.x1):]
        # Set outer diagonal elements to offd
        L[..., self.x1, self.x2] = offd
        # Set diagonal elements to di (real)
        d = torch.diagonal(L, dim1=-2, dim2=-1)
        d[()] = di
        # perform L @ L.T.conj()
        return L @ torch.transpose(L, -1, -2).conj()
    
##############################################################################
# Matrix distances (should be moved from here to a better place)
##############################################################################
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
    
class RevisedWishartLoss(object):
    def __init__(self, eps: float = 1e-5) -> None:
        if not (eps >= 0):
            raise ValueError("eps should be greater or equal to 0")
        self.eps = eps
    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Regularization of cov matrices with the given eps
        ri = self.eps * torch.eye(A.shape[-1], device=A.device)
        Ar = A + ri
        Br = B + ri
        _, ldetA = torch.linalg.slogdet(Ar)
        _, ldetB = torch.linalg.slogdet(Br)
        # Second trace
        d_tr = torch.sum(torch.diagonal(torch.linalg.solve(Br, Ar).real, dim1=-2, dim2=-1), dim=-1)
        # Final result
        #return (ldetB.mean() - ldetA.mean() + d_tr.mean()) - A.shape[-1]
        return (ldetB - ldetA + d_tr) - A.shape[-1]
    
class WishartLoss(object):
    def __init__(self, eps: float = 1e-5) -> None:
        if not (eps >= 0):
            raise ValueError("eps should be greater or equal to 0")
        self.eps = eps
    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Regularization of cov matrices with the given eps
        ri = self.eps * torch.eye(A.shape[-1], device=A.device)
        Ar = B + ri
        Br = A + ri
        _, ldetB = torch.linalg.slogdet(Br)
        # Second trace
        d_tr = torch.sum(torch.diagonal(torch.linalg.solve(Br, Ar).real, dim1=-2, dim2=-1), dim=-1).mean()
        # Final result
        return (ldetB.mean() + d_tr) - A.shape[-1]
    
class SymmetricRevisedWishartLoss_RelPreload(object):
    def __init__(self, eps: float = 1e-5, rel_eps: float = 1e-5) -> None:
        if not (eps >= 0):
            raise ValueError("eps should be greater or equal to 0")
        if not (rel_eps >= 0):
            raise ValueError("rel_eps should be greater or equal to 0")
        self.eps = eps
        self.reps = rel_eps
    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Regularization of cov matrices with the given reps
        di = torch.einsum("...ii->...", A).real + torch.einsum("...ii->...", B).real
        ri = (self.eps + self.reps * di[..., None, None]) * torch.eye(A.shape[-1], device=A.device)
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
    
class FrobeniusNormRelativeMeanSquaredLoss(object):
    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return (torch.linalg.norm(A-B, ord='fro', dim=(-2,-1)).pow(2) /
                torch.linalg.norm(B, ord='fro', dim=(-2,-1)).pow(2)).mean()

