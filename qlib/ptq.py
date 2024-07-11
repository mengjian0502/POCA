"""
Post training quantization
"""

import torch as th
from torch.nn.common_types import _size_2_t
from .base import QBase, QConv2d, QLinear
import torch.nn.functional as thf

def round_ste(x:th.Tensor):
    """
    Straight through estimator
    """
    return (x.round() - x).detach() + x

def lp_loss(pred, target, p=2.0, reduction='none'):
    """
    loss function measured in lp norm
    """
    if reduction == 'none':
        return (pred-target).abs().pow(p).sum(1).mean()
    else:
        return (pred-target).abs().pow(p).mean()
    
class AdaRound(QBase):
    """
    Weight quantizer: Up or Down? Adaptive Rounding for Post-Training Quantization
    
    https://arxiv.org/abs/2004.10568
    """
    def __init__(self, nbit: int = 8, train_flag: bool=True, weights: th.Tensor=None) -> None:
        super().__init__(nbit)
        self.iter = 0
        self.train_flag = train_flag

        self.register_buffer("lb", weights.min())
        self.register_buffer("ub", weights.max())

        # integer boundary
        self.qlb = (-(1 << (self.nbit-1)))
        self.qub = ((1 << (self.nbit-1)) - 1)

        # initialize the alpha
        self.init_flag = True

        # parameters
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3

        # register the learnable parameter
        self.register_alpha(weights)

    def register_alpha(self, x:th.Tensor):
        xfloor = x.div(self.scale).floor()

        # compute alpha
        diff = x.div(self.scale).sub(xfloor)
        alpha = -th.log((self.zeta-self.gamma) / (diff - self.gamma) - 1)
        self.register_parameter("alpha", th.nn.Parameter(alpha))

    def get_qparam(self, x:th.Tensor):
        lb = th.min(self.lb, x.min())
        ub = th.max(self.ub, x.max())

        # update boundary
        self.lb = lb.clone()
        self.ub = ub.clone()

    def h(self):
        return th.clamp(th.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    
    def q(self, x:th.Tensor):
        scale = self.ub.sub(self.lb).div(self.qub - self.qlb)
        offset = self.qlb - self.lb / scale

        self.scale.copy_(scale)
        self.offset.copy_(offset)

        if self.init_flag:
            self.register_alpha(x)
            self.init_flag = False

        # quantization
        xfloor = x.div(self.scale).floor()
        soft_shift = self.h()

        # quantize
        if self.train_flag:
            xint = xfloor + soft_shift
        else:
            xint = xfloor + self.alpha.ge(0.0).float()

        xq = xint + self.offset
        out = th.clamp(xq, self.qlb, self.qub)

        # dequantize
        out = out.sub(self.offset).mul(self.scale)
        return out
    
    def train_func(self, x: th.Tensor):
        self.get_qparam(x)
        return super().train_func(x)

    def eval_func(self, x: th.Tensor):
        xq = self.q(x)
        return xq
    
class LearningQ(QBase):
    """
    Activation quantizer with fully learnable scaling factor

    The learnable scaling factor is initialized as the optimal value of the first batch
    """
    def __init__(self, nbit: int = 8, train_flag: bool = True):
        super().__init__(nbit)
        self.train_flag = train_flag

        # register learnable parameter 
        self.register_parameter("delta", th.nn.Parameter(th.tensor(1.0)))
        
        # initialization flag
        self.initialize = False

        self.prev_int_frame = None
        self.round_err = 0.0
        self.frame_err_all = []

    def compute_frame_err(self, xq:th.Tensor):
        if self.prev_int_frame is None:
            self.prev_int_frame = xq
        else:
            self.round_err = xq.sub(self.prev_int_frame).abs()
            self.prev_int_frame = xq

            self.frame_err_all.append(self.round_err)
    
    def get_fp_range(self, x:th.Tensor):
        y = th.flatten(x, start_dim=1)
        batch_min = th.min(y, 1)[0].mean()
        batch_max = th.max(y, 1)[0].mean()
        return batch_min, batch_max
    
    def quantize(self, x:th.Tensor, xmin, xmax):
        delta = (xmax - xmin) / (2 ** self.nbit - 1)
        zero_point = (-xmin / delta).round()

        xint = th.round(x / delta)
        xq = th.clamp(xint + zero_point, 0, 2**self.nbit - 1)
        xdq = (xq - zero_point) * delta
        return xdq
    
    def initialize_qparam(self, x:th.Tensor):
        """
        Find the optimal scaling factor in the first batch
        """
        x_min, x_max = self.get_fp_range(x)
        best_loss = 1e+10

        for i in range(80):
            new_max = x_max * (1.0 - (i * 0.01))
            new_min = x_min * (1.0 - (i * 0.01))

            # quantize and dequantize for mse 
            xdq = self.quantize(x, new_min, new_max)
            loss = lp_loss(xdq, x, p=2.4, reduction='all')

            if loss < best_loss:
                best_loss = loss
                delta = (new_max - new_min) / (2**self.nbit - 1)
                zero_point = (-new_min / delta).round()
        
        return delta, zero_point

    def q(self, x:th.Tensor):
        if not self.initialize:
            if self.train_flag:
                delta, zero_point = self.initialize_qparam(x)
                self.delta.data = delta
                self.offset.data = zero_point

                self.initialize = True

        # quantize
        xr = round_ste(x / self.delta) + self.offset
        xq = th.clamp(xr, min=0.0, max=2**self.nbit-1)

        # dequantize
        xdq = (xq - self.offset) * self.delta
        return xdq
    
    def extra_repr(self) -> str:
        return super().extra_repr() + f", delta={self.delta.data.item():.4e}"
    


