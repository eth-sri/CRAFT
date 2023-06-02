import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from functools import reduce
from ai.ai_util import AbstractElement
import ai.concrete_layers as concrete_layers
from typing import Optional, List, Tuple, Union
from ai.concrete_layers import Bias

class AbstractModule(nn.Module):
    def __init__(self, save_bounds=True):
        super(AbstractModule, self).__init__()
        self.save_bounds = save_bounds
        self.bounds = None
        self.dim = None

    def update_bounds(self, bounds, detach=True):
        lb, ub = bounds

        if detach:
            lb, ub = lb.detach(), ub.detach()

        if self.dim is not None:
            lb = lb.view(-1, *self.dim)
            ub = ub.view(-1, *self.dim)

        if self.bounds is None:
            self.bounds = (lb, ub)
        else:
            self.bounds = (torch.maximum(lb, self.bounds[0]), torch.minimum(ub, self.bounds[1]))

    def reset_bounds(self):
        self.bounds = None

    def reset_dim(self):
        self.dim = None

    def set_dim(self, x):
        self.dim = torch.tensor(x.shape[1:])
        return self.forward(x)


class Sequential(AbstractModule):
    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.dim = None

    @classmethod
    def from_concrete_network(
            cls,
            network: nn.Sequential,
            input_dim: Tuple[int, ...],
    ) -> "Sequential":
        abstract_layers: List[AbstractModule] = []
        for i, layer in enumerate(network.children()):
            if i == 0:
                current_layer_input_dim = input_dim
            else:
                current_layer_input_dim = abstract_layers[-1].output_dim

            if isinstance(layer, nn.Sequential):
                abstract_layers.append(Sequential.from_concrete_network(layer, current_layer_input_dim))
            elif isinstance(layer, nn.Linear):
                abstract_layers.append(Linear.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, concrete_layers.Bias):
                abstract_layers.append(Bias.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, nn.ReLU):
                abstract_layers.append(ReLU.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, nn.Conv2d):
                abstract_layers.append(Conv2d.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, nn.Flatten):
                abstract_layers.append(Flatten.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, nn.AvgPool2d):
                abstract_layers.append(AvgPool2d.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, nn.Identity):
                abstract_layers.append(Identity(current_layer_input_dim))
            elif isinstance(layer, nn.BatchNorm2d):
                abstract_layers.append(BatchNorm2d.from_concrete_layer(layer, current_layer_input_dim))
            else:
                raise NotImplementedError(f"Unsupported layer type: {type(layer)}")
        return Sequential(*abstract_layers)

    def forward_between(self, i_from, i_to, x):
        for layer in self.layers[i_from:i_to]:
            if isinstance(x, AbstractElement) and layer.save_bounds:
                layer.update_bounds(x.concretize(), detach=True)
            x = layer(x)
        return x

    def forward_until(self, i, x):
        return self.forward_between(0, i+1, x)

    def forward_from(self, i, x):
        return self.forward_between(i+1, len(self.layers), x)

    def forward(self, x):
        return self.forward_from(-1, x)

    def reset_bounds(self, i_from=0, i_to=-1):
        self.bounds = None
        i_to = i_to+1 if i_to != -1 else len(self.layers)
        for layer in self.layers[i_from:i_to]:
            layer.reset_bounds()

    def reset_dim(self, i_from=0, i_to=-1):
        i_to = i_to+1 if i_to != -1 else len(self.layers)
        for layer in self.layers[i_from:i_to]:
            layer.reset_dim()

    def set_dim(self, x):
        self.dim = torch.tensor(x.shape[1:])
        for layer in self.layers:
            x = layer.set_dim(x)
        return x

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]


class Conv2d(nn.Conv2d, AbstractModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 dim=None):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.dim = dim

    def forward(self, x):
        if isinstance(x, AbstractElement):
            return x.conv2d(self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return super(Conv2d, self).forward(x)

    @classmethod
    def from_concrete_layer(
        cls, layer: nn.Conv2d, input_dim: Tuple[int, ...]
    ) -> "Conv2d":
        abstract_layer = cls(
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
            layer.bias is not None,
        )
        abstract_layer.weight.data = layer.weight.data
        if layer.bias is not None:
            abstract_layer.bias.data = layer.bias.data

        abstract_layer.output_dim = abstract_layer.getShapeConv(input_dim)

        return abstract_layer

    def getShapeConv(self, input_dim):
        inChan, inH, inW = input_dim
        kH, kW = self.kernel_size

        outH = 1 + int((2 * self.padding[0] + inH - kH) / self.stride[0])
        outW = 1 + int((2 * self.padding[1] + inW - kW) / self.stride[1])
        return (self.out_channels, outH, outW)


class ReLU(nn.ReLU, AbstractModule):
    def __init__(self, dim: Optional[Tuple]=None) -> None:
        super(ReLU, self).__init__()
        self.deepz_lambda = nn.Parameter(-torch.ones(dim, dtype=torch.float))

    def get_neurons(self) -> int:
        return reduce(lambda a, b: a * b, self.dim)

    def forward(self, x) -> Union[AbstractElement,Tensor]:
        if isinstance(x, AbstractElement):
            out, deepz_lambda = x.relu(self.deepz_lambda, self.bounds)
            if deepz_lambda is not None and (self.deepz_lambda < 0).any():
                self.deepz_lambda.data = deepz_lambda
            return out
        else:
            return super(ReLU, self).forward(x)

    @classmethod
    def from_concrete_layer(
        cls, layer: nn.ReLU, input_dim: Tuple[int, ...]
    ) -> "ReLU":
        abstract_layer = cls(input_dim)
        abstract_layer.output_dim = input_dim
        return abstract_layer

class Identity(nn.Identity, AbstractModule):
    def __init__(self, input_dim: Tuple[int, ...]) -> None:
        super(Identity, self).__init__()
        self.output_dim = input_dim

    def forward(self, x: Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        return x

class Flatten(AbstractModule):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x: Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        return x.view((x.size()[0], -1))
    
    @classmethod
    def from_concrete_layer(
        cls, layer: nn.Flatten, input_dim: Tuple[int, ...]
    ) -> "Flatten":
        abstract_layer = cls()
        abstract_layer.output_dim = input_dim
        return abstract_layer


class Linear(nn.Linear,AbstractModule):
    def __init__(self, in_features:int, out_features:int, bias:bool=True) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)

    def forward(self, x:Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            return x.linear(self.weight, self.bias)
        return super(Linear, self).forward(x)

    @classmethod
    def from_concrete_layer(
        cls, layer: nn.Linear, input_dim: Tuple[int, ...]
    ) -> "Linear":
        abstract_layer = cls(
            layer.in_features,
            layer.out_features,
            layer.bias is not None,
        )
        abstract_layer.weight.data = layer.weight.data
        if layer.bias is not None:
            abstract_layer.bias.data = layer.bias.data

        abstract_layer.output_dim = input_dim

        return abstract_layer


class _BatchNorm(nn.modules.batchnorm._BatchNorm, AbstractModule):
    def __init__(self, out_features:int, dimensions:int, affine:bool=False):
        super(_BatchNorm, self).__init__(out_features, affine=affine)
        # self.running_mean = None
        # self.running_var = None
        self.current_mean = None
        self.current_var = None
        self.affine = affine
        if not self.affine:
            self.weight = 1
            self.bias = 0
        if dimensions == 1:
            self.mean_dim = [0]
            self.view_dim = (1, -1)
        if dimensions == 2:
            self.mean_dim = [0, 2, 3]
            self.view_dim = (1, -1, 1, 1)

    def forward(self, x:Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            return x.batch_norm(self)
        if self.training:
            momentum = 1 if self.momentum is None else self.momentum
            self.current_mean = x.mean(dim=self.mean_dim).detach()
            self.current_var = x.var(unbiased=False, dim=self.mean_dim).detach()
            if self.track_running_stats:
                if self.running_mean is not None and self.running_var is not None:
                    self.running_mean = self.running_mean * (1 - momentum) + self.current_mean * momentum
                    self.running_var = self.running_var * (1 - momentum) + self.current_var * momentum
                else:
                    self.running_mean = self.current_mean
                    self.running_var = self.current_var
        else:
            self.current_mean = self.running_mean
            self.current_var = self.running_var
        c = (self.weight / torch.sqrt(self.current_var + self.eps))
        b = (-self.current_mean * c + self.bias)
        return x*c.view(self.view_dim)+b.view(self.view_dim)

    @classmethod
    def from_concrete_layer(
        cls, layer: Union[nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d], input_dim: Tuple[int, ...]
    ) -> "_BatchNorm":
        abstract_layer = cls(
            layer.num_features,
            layer.affine,
        )
        abstract_layer.running_var.data = layer.running_var.data
        abstract_layer.running_mean.data = layer.running_mean.data
        if layer.affine:
            abstract_layer.weight.data = layer.weight.data
            abstract_layer.bias.data = layer.bias.data

        abstract_layer.track_running_stats = layer.track_running_stats
        abstract_layer.training = False
        abstract_layer.output_dim = input_dim

        return abstract_layer


class BatchNorm1d(_BatchNorm):
    def __init__(self, out_features:int, affine:bool=False):
        super(BatchNorm1d, self).__init__(out_features, 1, affine)


class BatchNorm2d(_BatchNorm):
    def __init__(self, out_features:int, affine:bool=False):
        super(BatchNorm2d, self).__init__(out_features, 2, affine)


class AvgPool2d(nn.AvgPool2d, AbstractModule):
    def __init__(self, kernel_size:int, stride:Optional[int]=None, padding:int=0):
        super(AvgPool2d, self).__init__(kernel_size, stride, padding)

    def forward(self, x:Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            assert self.padding == 0
            return x.avg_pool2d(self.kernel_size, self.stride)
        return super(AvgPool2d, self).forward(x)

    @classmethod
    def from_concrete_layer(
        cls, layer: Union[nn.AvgPool2d], input_dim: Tuple[int, ...]
    ) -> "AvgPool2d":
        abstract_layer = cls(
            layer.kernel_size,
            layer.stride,
            layer.padding,
        )
        abstract_layer.output_dim = input_dim

        return abstract_layer


class GlobalAvgPool2d(nn.AdaptiveAvgPool2d, AbstractModule):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__(1)

    def forward(self, x: Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            return x.global_avg_pool2d()
        return super(GlobalAvgPool2d, self).forward(x)


class Bias(AbstractModule):
    def __init__(self, bias=0, fixed=False):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(bias*torch.ones(1))
        self.bias.requires_grad_(not fixed)

    def forward(self, x:Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        return x + self.bias

    @classmethod
    def from_concrete_layer(
        cls, layer: Union[Bias], input_dim: Tuple[int, ...]
    ) -> "Bias":
        abstract_layer = cls(layer.bias)
        abstract_layer.output_dim = input_dim
        return abstract_layer


class Scale(AbstractModule):
    def __init__(self, scale=1, fixed=False):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(scale*torch.ones(1))
        self.scale.requires_grad_(not fixed)

    def forward(self, x: Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        return x * self.scale


class Normalization(AbstractModule):
    def __init__(self, mean, sigma):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(mean)
        self.sigma = nn.Parameter(sigma)
        self.mean.requires_grad_(False)
        self.sigma.requires_grad_(False)

    def forward(self, x: Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        target_shape = [1,-1] + (x.dim()-2) * [1]
        if isinstance(x, AbstractElement):
            return x.normalize(self.mean.view(target_shape), self.sigma.view(target_shape))
        return (x - self.mean.view(target_shape)) / self.sigma.view(target_shape)


def add_bounds(lidx, zono, bounds=None, layer=None):
    lb_new, ub_new = zono.concretize()
    if layer is not None:
        if layer.bounds is not None:
            lb_old, ub_old = layer.bounds
            lb_new, ub_new = torch.max(lb_old, lb_new).detach(), torch.min(ub_old, ub_new).detach()
        layer.bounds = (lb_new, ub_new)
    if bounds is not None:
        bounds[lidx] = (lb_new, ub_new)
        return bounds