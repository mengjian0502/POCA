"""
POCA: Post-Training Quantization Caliborator
"""
import os 
import torch as th
import torch.nn.functional as thf
from tqdm import tqdm
from typing import Union, List
import nvdiffrast.torch as dr

from qlib.base import QConvTranspose2dWN, QLinearWN
from qlib.ptq import AdaRound, LearningQ
from qlib.utils import DataSaverHook, AverageMeter
from qlib.qwrap import model2ptq, _parent_name
from .utils import plot_masks

class UVFetcher:
    """
    Modified on top of the default Renderer
    """
    def __init__(self):
        self.glctx = dr.RasterizeGLContext()

    def render(self, M, pos, pos_idx, uv, uv_idx, tex, resolution=[2048, 1334]):
        ones = th.ones((pos.shape[0], pos.shape[1], 1)).to(pos.device)
        pos_homo = th.cat((pos, ones), -1)
        projected = th.bmm(M, pos_homo.permute(0, 2, 1))
        projected = projected.permute(0, 2, 1)
        proj = th.zeros_like(projected)
        proj[..., 0] = (
            projected[..., 0] / (resolution[1] / 2) - projected[..., 2]
        ) / projected[..., 2]
        proj[..., 1] = (
            projected[..., 1] / (resolution[0] / 2) - projected[..., 2]
        ) / projected[..., 2]
        clip_space, _ = th.max(projected[..., 2], 1, keepdim=True)
        proj[..., 2] = projected[..., 2] / clip_space

        pos_view = th.cat(
            (proj, th.ones(proj.shape[0], proj.shape[1], 1).to(proj.device)), -1
        )
        pos_idx_flat = pos_idx.view((-1, 3)).contiguous()
        uv_idx = uv_idx.view((-1, 3)).contiguous()
        # tex = tex.permute((0, 2, 3, 1)).contiguous()

        rast_out, rast_out_db = dr.rasterize(
            self.glctx, pos_view, pos_idx_flat, resolution
        )
        texc, _ = dr.interpolate(uv, rast_out, uv_idx)
        return texc


class PTQTrainer(object):
    """
    Trainer for post training quantization
    """
    def __init__(self, model:th.nn.Module, max_iter:int=100, dataloader=None, args=None, 
                texmean=None, texstd=None, vertmean=None, vertstd=None, logger=None) -> None:
        self.model = model
        self.logger = logger

        # args
        self.args = args

        # precision
        self.wbit = args.wbit
        self.abit = args.abit

        # dataloader for clibration
        self.dataloader = dataloader

        # max epochs
        self.max_iter = max_iter

        # tau
        self.tau = args.tau
        
        # visibilitiy map
        self.uvfetcher = UVFetcher()
        self.frame_size = 1024

        # for renderer
        self.texmean = texmean
        self.texstd = texstd
        self.vertmean = vertmean
        self.vertstd = vertstd

        # for mask visualization
        self.mask_path = os.path.join(self.args.result_path, "mask_1024x1024.png")


    def forward_model(self, data):
        M = data["M"].cuda()
        vert_ids = data["vert_ids"].cuda()
        uvs = data["uvs"].cuda()
        uv_ids = data["uv_ids"].cuda()
        avg_tex = data["avg_tex"].cuda()
        view = data["view"].cuda()
        verts = data["aligned_verts"].cuda()
        cams = data["cam"].cuda()

        output = {}

        if self.args.arch == "warp":
            pred_tex, pred_verts, unwarped_tex, warp_field, kl = self.model(
                avg_tex, verts, view, cams=cams
            )
            output["unwarped_tex"] = unwarped_tex
            output["warp_field"] = warp_field
        else:
            pred_tex, pred_verts, kl = self.model(avg_tex, verts, view, cams=cams)

        pred_verts = pred_verts * self.vertstd + self.vertmean
        pred_tex = (pred_tex * self.texstd + self.texmean) / 255.

        # renderer
        uv_mask = self.uvfetcher.render(M, pred_verts, vert_ids, uvs, uv_ids, pred_tex, self.args.resolution)

        output["pred_verts"] = pred_verts
        output["pred_tex"] = pred_tex
        output["uv"] = uv_mask
        
        # for decoder model reconstruction
        output["view"] = view

        return output

    def fetch_layer_data(self, layer: Union[QConvTranspose2dWN, QLinearWN], batch, model_calib=False):
        hook = DataSaverHook(store_input=True, store_output=True)
        handle = layer.register_forward_hook(hook)

        with th.no_grad():
            output = self.forward_model(batch)

        handle.remove()
        if not model_calib:
            return hook.input[0], hook.output, output["uv"].detach()
        else:
            return hook.input[0], output["uv"].detach(), hook.output, output["uv"].detach()

        
    def uv2idx(self, uv:th.Tensor):
        """
        Conver the uv grid into the indexes

        Args:
        - uv (Tensor): vt_img that represents the visibility map 

        Output:
        - Indexes that generated from the normalized visibility map (grid)
        """

        idx = uv.mul(self.frame_size-1).round().clamp_max(self.frame_size-1)
        return idx.int()

    def mask_indexing(self, uv:th.Tensor):
        """
        Highlighting the pixels based on indexes
        """
        mask = th.zeros(1024, 1024).cuda()

        uvf = uv.view(-1, 2)
        uvx = uvf[:, 0]
        uvy = uvf[:, 1]

        mask[uvy, uvx] = 1.0
        return mask
    
    def uv2masks(self, ind:th.Tensor):
        masks = []
        for idx in ind:
            mask = self.mask_indexing(idx)
            masks.append(mask.unsqueeze(0))
        
        masks = th.cat(masks, dim=0)
        return masks
    
    def fetch_layer_data_all(self, layer: Union[QConvTranspose2dWN, QLinearWN], layer_name):
        cached_batches = []
        self.uv_masks = []

        for i, batch in enumerate(tqdm(self.dataloader)):
            x, y, uv = self.fetch_layer_data(layer, batch)

            ind = self.uv2idx(uv)
            masks = self.uv2masks(ind)
            self.uv_masks.append(masks.unsqueeze(1))

            cached_batches.append((x, y))
        
        plot_masks(masks, path=self.mask_path)
        self.logger.info(f"Data Fetched for layer {layer_name}! | last batch of mask: {self.mask_path}")
        th.cuda.empty_cache()
        return cached_batches
    
    def fetch_model_data_all(self, model:th.nn.Module, layer_name):
        cached_batches = []
        self.uv_masks = []

        for i, batch in enumerate(tqdm(self.dataloader)):
            x, view, y, uv = self.fetch_layer_data(model, batch, model_calib=True)

            ind = self.uv2idx(uv)
            masks = self.uv2masks(ind)
            self.uv_masks.append(masks.unsqueeze(1))
            cached_batches.append(((x, view), y))
        
        plot_masks(masks, path=self.mask_path)
        self.logger.info(f"Data Fetched for layer {layer_name}! | last batch of mask: {self.mask_path}")
        th.cuda.empty_cache()
        return cached_batches

    
    def downsample_mask(self, uv:th.Tensor, height):
        # stride and window
        kernel = uv.size(2) // height
        mask = thf.max_pool2d(uv, kernel_size=kernel, stride=kernel)
        return mask.cuda()
    
    def shape_wise_filter(self, y:th.Tensor):
        """
        Pixel filter for calibration:
        Filter out the 1x1xC pixels that has the long-tailed distribution

        The shape-wise std scores are considered as the metric for filtering
        The sparsity of the masking is controlled by tau. 
        """
        assert len(y.size()) == 4, "The shape of the feature map tensor must be 4-D"

        num_pixel_keep = int(y.size(2)*y.size(3) * self.tau)

        b, c, h, w = y.size()
        yd = y.detach()

        with th.no_grad():
            masks = []

            # row channel plane
            ystd = yd.std(dim=[1])

            for i in range(b):
                mask = th.zeros(h, w)
                scores = ystd[i]

                # top-k score
                tpk_score = th.topk(scores.view(-1), num_pixel_keep, sorted=True)
                threshold = tpk_score.values[-1]

                # mask[scores.lt(threshold), :] = 1.0
                mask = scores.lt(threshold).float()
                masks.append(mask.unsqueeze(0).unsqueeze(0))
        
        return th.cat(masks, dim=0).cuda()
    
    def row_wise_filter(self, y:th.Tensor):
        """
        Pixel filter for calibration:
        Filter out the row-channel pixels that has the long-tailed distribution

        The std scores of the row-channel plane are considered as the metric for filtering
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


    def layer_reconstruction(self, layer:Union[QConvTranspose2dWN, QLinearWN], layer_name, lr, cached_data:List):
        qparam = []

        # freeze the weights and bias
        layer.weight.requires_grad = False
        layer.bias.requires_grad = False
        layer.g.requires_grad = False

        # assign the quantizer 
        layer.wq = AdaRound(nbit=self.wbit, train_flag=True, weights=layer.weight).cuda()
        layer.xq = LearningQ(nbit=self.abit, train_flag=True).cuda()
        
        qparam += [layer.wq.alpha, layer.xq.delta]

        # optimizer
        optimizer = th.optim.Adam(qparam, lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
        
        # scheduler
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_iter, eta_min=0.)

        # loss function
        loss_fn = th.nn.MSELoss()

        # calibration
        if isinstance(layer, (QConvTranspose2dWN, QLinearWN)):
            loss = AverageMeter()
            spars = AverageMeter()
            pbar = tqdm(range(self.max_iter))
            for e in pbar:
                for idx, batch in enumerate(cached_data):
                    x, y = batch 

                    # cuda 
                    x = x.cuda()
                    y = y.cuda()

                    # forward pass 
                    out = layer(x)

                    # visibility masks
                    if isinstance(layer, QConvTranspose2dWN):
                        if self.tau > 0:
                            visib = self.uv_masks[idx]
                            ymask = self.downsample_mask(visib, height=y.size(2))

                            # std filtering
                            # row_mask = self.row_wise_filter(y)
                            filter_mask = self.shape_wise_filter(y)
                            ymask = ymask.mul(filter_mask)

                            # sparsity
                            s = ymask[ymask.eq(0.)].numel() / ymask.numel()
                            spars.update(s)

                            y = y.mul(ymask)
                            out = out.mul(ymask)
                        else:
                            s = 0.0
                    else:
                        # no sparsity on linear layer
                        s = 0.0

                    rec_loss = loss_fn(out, y)
                    loss.update(rec_loss.item())

                    optimizer.zero_grad()
                    rec_loss.backward(retain_graph=True)
                    optimizer.step()
                
                scheduler.step()
                pbar.set_description(f"Rec loss:{layer_name} = {loss.avg:.4e} | yspars = {s:.3f} | tau = {self.tau}")

        self.logger.info(f"{layer_name} Done!")
        filename = f"{layer_name}_ymask.png"
        mask_path = os.path.join(self.args.result_path, filename)
        
        if isinstance(layer, QConvTranspose2dWN):
            if self.tau > 0:
                plot_masks(ymask.squeeze(1), mask_path)

        # switch mode
        layer.xq.train_flag = False
        layer.wq.train_flag = False
        th.cuda.empty_cache()

        return layer

    def nnfreeze(self, module:th.nn.Module):
        """
        Freeze the vanilla model before assigning the quantizers
        """
        for n, p in module.named_parameters():
            if p.requires_grad:
                p.requires_grad = False
        
    def dispersion_coef(self, dec_out:th.Tensor, mask:th.Tensor, output_path):
        assert len(dec_out.size()) == 4, "the feature map size must be 4-D"

        with th.no_grad():
            dec_out = dec_out.mul(mask)

            # batch-wise variance and mean
            std = dec_out.std(dim=0)

        th.save(std.detach().cpu(), output_path)
    
    def batch_kldiv(self, out:th.Tensor, y:th.Tensor):
        osfmx = thf.softmax(out, dim=0)
        ysfmx = thf.softmax(y, dim=0)

        kl_loss = thf.kl_div(osfmx, ysfmx, reduction="none")
        return kl_loss.mean()

    def model_reconstruction(self, model:th.nn.Module, layer_name:str, lr:float, cached_data:List):
        optimizer = th.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

        # scheduler
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_iter, eta_min=0.)

        # loss function
        loss_fn = th.nn.MSELoss()

        # meters
        loss = AverageMeter()
        spars = AverageMeter()

        # previous batch mean
        self.prev_b = 0

        for e in tqdm(range(self.max_iter)):
            for idx, batch in enumerate(cached_data):
                x, y = batch

                # cuda 
                x = x.cuda()
                y = y.cuda()

                # forward pass 
                out = model(x)
                
                if self.tau > 0:
                    # visibility masks
                    visib = self.uv_masks[idx]
                    ymask = self.downsample_mask(visib, height=y.size(2))

                    # swd filtering
                    filter_mask = self.shape_wise_filter(y)
                    ymask = ymask.mul(filter_mask)

                    # sparsity
                    s = ymask[ymask.eq(0.)].numel() / ymask.numel()

                    y = y.mul(ymask)
                    out = out.mul(ymask)
                else:
                    s = 0.0
                    
                spars.update(s)

                mse_loss = loss_fn(out, y)
                rec_loss = mse_loss

                loss.update(rec_loss.item())

                optimizer.zero_grad()
                rec_loss.backward()
                optimizer.step()
            
            scheduler.step()
        
            self.logger.info(f"[{e+1} / {self.max_iter}] lr={scheduler.get_last_lr()[0]:.4e} | Rec loss:{layer_name} = {loss.avg:.4e} | yspars = {s:.3f}")

        return model

    def fit(self):
        # convert the vanilla modules to quantization-ready modules
        tex_dec = self.model.module.dec
        
        # freeze all the weights and bias from learning
        self.nnfreeze(tex_dec)

        qtex_dec = model2ptq(tex_dec)

        # vanilla model copy
        vanilla_dec = qtex_dec.texture_decoder
        
        # insert the low precision decoder back
        self.model.module.dec = qtex_dec
        modules = dict(qtex_dec.named_modules(remove_duplicate=False))

        # map to cuda
        self.model = self.model.cuda()

        # layer-wise calibration, fetch layer-wise data for less memory consumption.
        for n, m in modules.items():
            if isinstance(m, (QConvTranspose2dWN)):
                if not "texture_fc" in n:
                    cached_data = self.fetch_layer_data_all(m, n)

                    new_layer = self.layer_reconstruction(m, n, lr=self.args.lr, cached_data=cached_data)
                    parent_name, name = _parent_name(n)

                    setattr(modules[parent_name], name, new_layer)
        
        # release memory
        del cached_data

        # model-wise calibration
        if self.args.model_calib:
            qtex = qtex_dec.texture_decoder

            # fetch the calibration data of the texture decoder 
            cached_dec_data = self.fetch_layer_data_all(vanilla_dec, "Texture_Decoder")
            ptq_tex = self.model_reconstruction(qtex, layer_name="Texture_Decoder", lr=self.args.lr, cached_data=cached_dec_data)
            
            qtex_dec.texture_decoder = ptq_tex
            self.model.module.dec = qtex_dec
        
        return self.model