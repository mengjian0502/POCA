"""
Base quanitzation module
"""

from typing import List, Optional, Union
import torch as th
from torch import Tensor
from torch.nn.common_types import _size_2_t
import torch.nn.functional as thf
from qlib.utils import AverageMeter

class QBase(th.nn.Module):
    """
    Base parent class for quantization method design
    
    Args:
    - nbit: int
    Precision of the quantization target (e.g., weight, activation, bias)

    deploy: bool
    Post quantization deployment
    """
    def __init__(self, nbit:int=8) -> None:
        super().__init__()
        self.nbit = nbit
        self.deploy = False

        self.register_buffer("scale", th.tensor(1.0))
        self.register_buffer("offset", th.tensor(0.0))

    def q(self, x:th.Tensor):
        return x
    
    def train_func(self, x:th.Tensor):
        return self.q(x)
    
    def eval_func(self, x:th.Tensor):
        return self.train_func(x)
    
    def forward(self, x:th.Tensor):
        if not self.deploy:
            y = self.train_func(x)
        else:
            y = self.train_func(x)
        return y
    
    def extra_repr(self) -> str:
        return super().extra_repr() + f"nbit={self.nbit}"
    
class QConv2d(th.nn.Conv2d):
    def __init__(self, in_channels: int, 
            out_channels: int, 
            kernel_size: _size_2_t, 
            stride: _size_2_t = 1, 
            padding: _size_2_t = 0, 
            dilation: _size_2_t = 1, 
            groups: int = 1, 
            bias: bool = True, 
            padding_mode: str = 'zeros', 
            device=None, 
            dtype=None
        ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        self.wbit = 8
        self.abit = 8

        # quantizer
        self.wq = th.nn.Identity()
        self.xq = th.nn.Identity()
        self.yq = th.nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        wq = self.wq(self.weight)
        xq = self.xq(x)
        
        # convolution
        output = thf.conv2d(
            xq, wq, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        output = self.yq(output)
        return output
    

class QLinear(th.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

        self.wbit = 8
        self.abit = 8

        # quantizer
        self.wq = th.nn.Identity()
        self.xq = th.nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        wq = self.wq(self.weight)
        xq = self.xq(input)

        output = thf.linear(xq, wq, self.bias)
        return output
    

class QConv2dWN(th.nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(QConv2dWN, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            True,
        )
        self.g = th.nn.Parameter(th.ones(out_channels))

        # quantizers
        self.wq = th.nn.Identity()
        self.xq = th.nn.Identity()

    def forward(self, x:th.Tensor):
        wnorm = th.sqrt(th.sum(self.weight**2))
        wn = self.weight * self.g[:, None, None, None] / wnorm,
        
        # quantize
        xq = self.xq(x)
        wq = self.wq(wn)

        return thf.conv2d(
            xq,
            wq,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
    
class QLinearWN(th.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(QLinearWN, self).__init__(in_features, out_features, bias)
        self.g = th.nn.Parameter(th.ones(out_features))

        # quantizers
        self.wq = th.nn.Identity()
        self.xq = th.nn.Identity()
    
    def forward(self, input):
        wnorm = th.sqrt(th.sum(self.weight**2))
        wn = self.weight * self.g[:, None] / wnorm

        # quantize
        xq = self.xq(input)
        wq = self.wq(wn)

        return thf.linear(xq, wq, self.bias)


class QConvTranspose2dWN(th.nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(QConvTranspose2dWN, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            True,
        )
        self.g = th.nn.Parameter(th.ones(out_channels))

        # quantizers
        self.wq = th.nn.Identity()
        self.xq = th.nn.Identity()

        # variance meter
        self.frame_diff = AverageMeter()
        self.prev_frame = None

    def forward(self, x):
        wnorm = th.sqrt(th.sum(self.weight**2))
        wn = self.weight * self.g[None, :, None, None] / wnorm

        # quantize
        xq = self.xq(x)
        wq = self.wq(wn)
        # xq = x
        # wq = wn

        # if self.prev_frame is None:
        #     self.prev_frame = xq.detach()
        # else:
        #     ferr = xq.sub(self.prev_frame).abs().mean()
        #     self.frame_diff.update(ferr.item())

        yq = thf.conv_transpose2d(
            xq,
            wq,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return yq
