"""
utils
"""

import torch 
import numpy as np
import matplotlib.pyplot as plt

def plot_masks(masks:torch.Tensor, path:str):
    if masks.size(0) > 1:
        fig, axs = plt.subplots(2,4, figsize=(15, 6), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.05)

        axs = axs.ravel()

        for i in range(8):
            axs[i].imshow(masks[i].detach().cpu().numpy())

        plt.savefig(path, dpi=300)
    
    else:
        plt.figure(figsize=(8,8))
        plt.imshow(masks[0,0].cpu().numpy())
        plt.savefig(path, dpi=300)
        