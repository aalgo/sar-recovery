from . import patches
from lib import io
from lib import sar
from lib import data
import numpy as np
from loguru import logger
import torch


smap = {'Shh': [0,0], 'Shv': [0,1], 
        'Svh': [1,0], 'Svv': [1,1]}

cmap = {"Shh2": (0,0),    "ShhShv*": (0,1), "ShhSvv*": (0,2),
        "ShvShh*": (1,0), "Shv2": (1,1),    "ShvSvv*": (1,2),
        "SvvSHH*": (2,0), "SvvShv*": (2,1), "Svv2":    (2,2)}


class ScatterCoherencePatchesDataset(patches.PatchesDataset):
        
    def __init__(self, 
                 base_path, 
                 date, 
                 patch_size, 
                 splitmask_fn_src, 
                 split, 
                 avg_window_size, 
                 scatter_elems=['Shh', 'Shv'],
                 coherence_elems=['Svv2']):

        if len(np.unique(scatter_elems)) != sum([i in smap.keys() for i in scatter_elems]):
            raise  ValueError(f"invalid elems '{scatter_elems}', allowed elems are '{list(smap.keys())}'")

        if len(np.unique(coherence_elems)) != sum([i in cmap.keys() for i in coherence_elems]):
            raise  ValueError(f"invalid elems '{coherence_elems}', allowed elems are '{list(cmap.keys())}'")

        self.base_path = base_path
        self.date = date
        self.splitmask_fn_src = splitmask_fn_src
        self.splitmask_fn = eval(splitmask_fn_src)
        self.scatter_elems = scatter_elems
        self.coherence_elems = coherence_elems
        self.avg_window_size = avg_window_size

        logger.info("loading scatter matrix")
        self.sm = io.load_bcn_scatter_matrix(base_path, date)

        logger.info("computing coherence matrix")
        self.cm = sar.compute_coherence_matrix(self.sm)

        splitmask = self.splitmask_fn(*self.sm.shape[:2])

        super().__init__(input_image = self.sm, 
                         output_image = self.cm, 
                         splitmask = splitmask, 
                         patch_size  = patch_size, 
                         split = split)

        logger.info(f"scatter   matrix shape is {self.sm.shape}, retrieving elems {[(k,smap[k]) for k in scatter_elems]}")
        logger.info(f"coherence matrix shape is {self.cm.shape}, retrieving elems {[(k,cmap[k]) for k in coherence_elems]}")

        self.avg_pool = sar.AvgPool2dComplex(n_channels=len(coherence_elems), kernel_size=avg_window_size)


    def __getitem__(self, idx):
        _r = super().__getitem__(idx)

        r = {}
        for k,v in _r.items():
            # rename patches
            if k == 'input_patch':
                k = 'scatter_patch'
            if k == 'output_patch':
                k = 'coherence_patch'
            r[k] = v

        scp = r['scatter_patch']
        r['scatter_patch'] = np.stack([scp[..., smap[e][0], smap[e][1]]  for e in self.scatter_elems], axis=-1)
        r['scatter_patch'] = np.transpose(r['scatter_patch'], [2,0,1])

        cmp = r['coherence_patch']
        r['coherence_patch'] = np.stack([cmp[..., cmap[e][0], cmap[e][1]]  for e in self.coherence_elems], axis=-1)
        r['coherence_patch'] = np.transpose(r['coherence_patch'], [2,0,1])

        r['avg_coherence_patch'] = self.avg_pool(torch.tensor(r['coherence_patch'])).numpy()

        return r




class ScatterPatchesDataset(patches.PatchesDataset):
        
    def __init__(self, 
                 base_path, 
                 date, 
                 patch_size, 
                 splitmask_fn_src, 
                 split, 
                 input_elems=['Svv', 'Svh'],
                 output_elems=['Shh']):

        if len(np.unique(input_elems)) != sum([i in smap.keys() for i in input_elems]):
            raise  ValueError(f"invalid elems '{input_elems}', allowed elemsn are '{list(smap.keys())}'")

        if len(np.unique(output_elems)) != sum([i in smap.keys() for i in output_elems]):
            raise  ValueError(f"invalid elems '{output_elems}', allowed elemsn are '{list(smap.keys())}'")

        self.base_path = base_path
        self.date = date
        self.splitmask_fn_src = splitmask_fn_src
        self.splitmask_fn = eval(splitmask_fn_src)
        self.input_elems = input_elems
        self.output_elems = output_elems

        logger.info("loading scatter matrix")
        self.sm = io.load_bcn_scatter_matrix(base_path, date)

        splitmask = self.splitmask_fn(*self.sm.shape[:2])

        super().__init__(input_image = self.sm, 
                         output_image = self.sm, 
                         splitmask = splitmask, 
                         patch_size  = patch_size, 
                         split = split)

        logger.info(f"scatter matrix shape is {self.sm.shape}, input elems {[(k,smap[k]) for k in input_elems]}, output elems {[(k,smap[k]) for k in output_elems]}")


    def __getitem__(self, idx):
        r = super().__getitem__(idx)

        scp = r['input_patch']
        r['input_patch'] = np.stack([scp[..., smap[e][0], smap[e][1]]  for e in self.input_elems], axis=-1)
        r['input_patch'] = np.transpose(r['input_patch'], [2,0,1])

        scp = r['output_patch']
        r['output_patch'] = np.stack([scp[..., smap[e][0], smap[e][1]]  for e in self.output_elems], axis=-1)
        r['output_patch'] = np.transpose(r['output_patch'], [2,0,1])

        return r