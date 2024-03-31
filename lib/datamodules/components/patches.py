from torch.utils.data import Dataset
import torch
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from rlxutils import subplots


TRAIN, TEST, VAL = 0,1,2
splitmap = {'train': TRAIN, 'test': TEST, 'val': VAL}
isplitmap = {v:k for k,v in splitmap.items()}

class PatchesDataset(Dataset):
    
    """
    creates a set of patches from an image so that they can be fed into a dataloader
    """
    
    def __init__(self, input_image, output_image, splitmask, patch_size, split):
        """
        image: an image of shape [h,w,...]
        splitmask: an array of shape [h,w] with values 0,1,2
        patch_size: the size of the patch
        split: the split to generate (train, test or val)
        """
        super().__init__()

        if not split in ['train', 'test', 'val']:
            raise ValueError(f"split must be 'train', 'test' or 'val' but got '{split}'")
        
        self.input_image = input_image
        self.output_image = output_image
        self.splitmask = splitmask
        self.patch_size = patch_size
        self.split = split
        
        if input_image.shape[:2] != splitmask.shape:
            raise ValueError(f"input image shape {input_image.shape} and splitmask shap {splitmask.shape} do not match")
        
        if output_image.shape[:2] != splitmask.shape:
            raise ValueError(f"output image shape {output_image.shape} and splitmask shap {splitmask.shape} do not match")

        if set(np.unique(splitmask)).intersection(set([TRAIN, TEST, VAL])) != set(np.unique(splitmask)):
            raise ValueError("invalid values in splitmask")
                                            
        # patch the splitmask and divide the patches in splits according to
        # the majority split pixels in each patch.
        self.patch_split = { k:[] for k in np.unique(splitmask) }
        h, w = self.splitmask.shape
        for y in range(0,h,patch_size):
            row = []
            for x in range(0,w,patch_size):
                patch = splitmask[y:y+patch_size, x:x+patch_size]

                # discard bordering patches
                if patch.shape!=(patch_size, patch_size):
                    continue

                # select the most frequent value (or randomly if more several values are most frequent)
                v, c = np.unique(patch, return_counts=True)
                most_freq_val = v[np.random.choice(np.argwhere(c==np.max(c))[:,0])]

                self.patch_split[most_freq_val].append((y,x))
                
            self.patches = self.patch_split[splitmap[split]]
        
        self.original_split_proportions = {k:(self.splitmask==v).sum()/np.product(self.splitmask.shape) for k,v in splitmap.items()}
        npatches = sum([len(v) for v in self.patch_split.values()])
        self.patch_split_proportions = {isplitmap[int(k)]: len(v)/npatches for k,v in self.patch_split.items()}
        
        # warn if the split proportions changed too much
        porig    = np.r_[[self.original_split_proportions[k] for k in ['train', 'test', 'val']]]
        ppatches = np.r_[[self.patch_split_proportions[k] for k in ['train', 'test', 'val']]]
        if np.mean(np.abs(porig-ppatches))>0.1:
            logger.warning(f"split proportions changed too much during patch split\n{self}")
        
        
    def __repr__(self):
        porig = "  ".join([f"{k}={v:.3f}" for k,v in self.original_split_proportions.items()])
        ppatches = "  ".join([f"{k}={v:.3f}" for k,v in self.patch_split_proportions.items()])
        r = f"""{self.__class__.__name__:26s}  patch_size={self.patch_size}  splitmask_dims={self.splitmask.shape}
original split proportions: {porig}
patch split proportions:    {ppatches}
        """
        return r
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        y,x = self.patches[idx]
        return {
                'input_patch':  self.input_image[y:y+self.patch_size, x:x+self.patch_size],
                'output_patch': self.output_image[y:y+self.patch_size, x:x+self.patch_size],
                'patch_coords': torch.tensor([y,x]).type(torch.int)
        }
        
    def plot_split(self):
        s = np.zeros(self.splitmask.shape) - 1
        for k,v in self.patch_split.items():
            for y,x in v:
                s[y:y+self.patch_size, x:x+self.patch_size] = k
        plt.imshow(s, interpolation="none")
        plt.colorbar()
        
    