"""
Layer converter
"""

import copy
import torch as th
from models import Conv2dWN, LinearWN, Conv2dWNUB, ConvTranspose2dWN
from .base import QConv2d, QLinearWN, QConvTranspose2dWN
from typing import List, Tuple

def get_module_device_and_dtype(module: th.nn.Module):
    parameters = list(module.parameters())

    if len(parameters) == 0:
        raise ValueError(f"Module {module.__class__.__qualname__} has no parameters!")
    
    devices = [p.device for p in parameters]
    types = [p.dtype for p in parameters]

    return devices[0], types[0]

def _parent_name(target: str) -> Tuple[str, str]:
    r = target.rsplit(".", 1)
    
    if len(r) == 1:
        return "", r[0]
    else:
        return r[0], r[1]

def wn2ptq(layer:th.nn.Module):
    with th.no_grad():
        layer = copy.deepcopy(layer)
        has_bias = layer.bias is not None

        device, dtype = get_module_device_and_dtype(layer)

        if type(layer) in [Conv2dWN, Conv2dWNUB]:
            new_layer = QConv2d(
                layer.in_channels, 
                layer.out_channels,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
                bias=has_bias,
                device=device, 
                dtype=dtype
            )

            # copy the weights
            new_layer.weight.data[:] = layer.weight

            if has_bias:
                new_layer.bias.data[:] = layer.bias

        elif type(layer) in [ConvTranspose2dWN]:
            new_layer = QConvTranspose2dWN(
                layer.in_channels,
                layer.out_channels, 
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
                bias=has_bias
            )

            # copy the weights
            new_layer.weight.data[:] = layer.weight
            new_layer.g.data[:] = layer.g

            if has_bias:
                new_layer.bias.data[:] = layer.bias
        
        elif type(layer) in [LinearWN]:
            new_layer = QLinearWN(
                layer.in_features,
                layer.out_features,
                bias=has_bias,
            )

            new_layer.weight.data[:] = layer.weight
            new_layer.g.data[:] = layer.g

            if has_bias:
                new_layer.bias.data[:] = layer.bias
    return new_layer

def model2ptq(model:th.nn.Module, inplace:bool = False):
    if not inplace:
        model = copy.deepcopy(model)

    modules = dict(model.named_modules(remove_duplicate=False))

    for layer_name, module in modules.items():
        if isinstance(module, (LinearWN, Conv2dWN, Conv2dWNUB, ConvTranspose2dWN)):
            new_module = wn2ptq(module)
            parent_name, name = _parent_name(layer_name)
            setattr(modules[parent_name], name, new_module)
    
    return model 

