"""
Calibrator of PTQ
"""

import torch as th
import torch.nn.functional as thf
from tqdm import tqdm
from typing import Union, List
from qlib.base import QConv2d, QLinear
from qlib.ptq import AdaRound, LearningQ
from qlib.utils import DataSaverHook, AverageMeter

class Calibrator(object):
    def __init__(self, decoder:th.nn.Module, max_iter:int=50, nsamples:int=1024, dataset=None, logger=None) -> None:
        self.decoder = decoder
        self.max_iter = max_iter

        # TODO: Move these arguments to ArgParse
        self.frame_size = 1024

        self.tau = 0.7
        
        self.logger = logger
        
        self.wbit = 8
        self.abit = 8

    def fetch_layer_data(self, layer: Union[QConv2d, QLinear], model:th.nn.Module, batch):
        hook = DataSaverHook(store_input=True, store_output=True)
        handle = layer.register_forward_hook(hook)

        with th.no_grad():
            output, loss = model(batch)
        
        handle.remove()
        return hook.input[0], hook.output
    
    def uv2idx(self, uv:th.Tensor):
        """
        Conver the uv grid into the indexes

        Args:
        - uv (Tensor): vt_img that represents the visibility map 

        Output:
        - Indexes that generated from the normalized visibility map (grid)
        """

        uv = uv.add(1.0).div(2.0)
        idx = uv.mul(self.frame_size-1).round()
        return idx.int()

    def mask_indexing(self, uv:th.Tensor):
        """
        Highlighting the pixels based on indexes
        """
        mask = th.zeros(self.frame_size, self.frame_size).cuda()

        uvf = uv.view(-1, 2)
        uvx = uvf[:, 0]
        uvy = uvf[:, 1]

        # TODO: visualize to make sure the orientation
        mask[uvy, uvx] = 1.0
        return mask
    
    def uv2masks(self, ind:th.Tensor):
        masks = []
        for idx in ind:
            mask = self.mask_indexing(idx)
            masks.append(mask)
        
        masks = th.cat(masks, dim=0)
        return masks
    
    def fetch_layer_data_all(self, layer: Union[QConv2d, QLinear], model:th.nn.Module, dataloader, layer_name):
        self.uv_masks = []
        cached_batches = []

        for i, batch in enumerate(tqdm(dataloader)):
            x, y = self.fetch_layer_data(layer, model, batch)

            if "tex" in layer_name:
                vt_img = None # TODO: check the grid of the sampling
                ind = self.uv2idx(vt_img)
                masks = self.uv2masks(ind)
                self.uv_masks.append(masks.unsqueeze(1))
            
            cached_batches.append((x, y))
        
        th.cuda.empty_cache()
        print(f"Data Fetched!")
        return cached_batches

    def downsample_mask(self, uv:th.Tensor, height):
        # stride and window
        kernel = uv.size(2) // height
        mask = thf.max_pool2d(uv, kernel_size=kernel, stride=kernel)
        return mask
    
    def row_wise_filter(self, y:th.Tensor):
        """
        Pixel filter for calibration:
        Filtering out the row-channel pixels that has the long-tailed distribution

        The std scores ofthe row-channel plane are considered as the metric for filtering
        The sparsity of the masking is controlled by tau. 

        Args:
        - tau: float, tunnable parameter that controls the intensity of filtering

        Output:
        - Feature map masks with the long-tailed distribution filtered. 
        """
        assert len(y.size()) == 4, "The shape of the feature map tensor must be 4-D"

        num_rows_keep = int(y.size(3) * self.tau)

        b, c, h, w = y.size()
        yd = y.detach()

        with th.no_grad():
            masks = []

            # row channel plane
            ystd = yd.std(dim=[1,3])

            for i in range(b):
                mask = th.zeros(h, w)
                scores = ystd[i]

                # top-k score
                tpk_score = th.topk(scores, num_rows_keep, sorted=True)
                threshold = tpk_score.values[-1]

                # activate rows
                mask[scores.lt(threshold), :] = 1.0
                masks.append(mask.unsqueeze(0).unsqueeze(0))
        
        return th.cat(masks, dim=0).cuda()

    def layer_reconstruction(self, layer:Union[QConv2d, QLinear], layer_name, lr, cached_data:List):
        qparam = []

        # assign the quantizer 
        layer.wq = AdaRound(nbit=self.wbit, train_flag=True, weights=layer.weight).cuda()
        layer.xq = LearningQ(nbit=self.abit, train_flag=True).cuda()

        qparam += [layer.wq.alpha, layer.xq.delta]

        # optimizer
        optimizer = th.optim.Adam(qparam, lr=lr)
        
        # scheduler
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_iter, eta_min=0.)

        # loss function
        loss_fn = th.nn.MSELoss()

        # calibration
        if isinstance(layer, (QConv2d, QLinear)):
            loss = AverageMeter()
            for e in tqdm(range(self.max_iter)):
                for idx, batch in enumerate(cached_data):
                    x, y = batch 

                    # cuda 
                    x = x.cuda()
                    y = y.cuda()

                    # forward pass 
                    out = layer(x)

                    # visibility masks
                    visib = self.uv_masks[idx]
                    visib = self.downsample_mask(visib, height=y.size(2))

                    # std filtering
                    row_mask = self.row_wise_filter(y)
                    ymask = visib.mul(row_mask)

                    y = y.mul(ymask)
                    out = out.mul(ymask)

                    rec_loss = loss_fn(out, y)
                    loss.update(rec_loss.item())

                    optimizer.zero_grad()
                    rec_loss.backward(retain_graph=True)
                    optimizer.step()
                
                scheduler.step()
            
            self.logger.info(f"[{e+1} / {self.max_iter}] lr={scheduler.get_last_lr()[0]:.4e} | Rec loss:{layer_name} = {loss.avg:.4e} | tau = {self.tau}")
        
        self.logger.info(f"{layer_name} Done!")

        return layer
            


        