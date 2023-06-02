"""
Based on HybridZonotope from DiffAI (https://github.com/eth-sri/diffai/blob/master/ai.py)
"""
import numpy as np
import torch
import torch.nn.functional as F
from .ai_util import clamp_image, head_from_bounds, AbstractElement
from monDEQ.mon import fft_conv
from typing import Optional, List, Tuple, Union, Dict
from torch import Tensor
import uuid
import pdb
from itertools import product
from time import time
import warnings

try:
    from gurobipy import GRB, Model, LinExpr, quicksum
    GUROBI_AVAILABLE = True
except:
    GUROBI_AVAILABLE = False
    warnings.warn("GUROBI not available, no inclusion checks for DeepPoly")



def map_dict(d, f):
    """ Maps a function over all entries in a zonotope-entry dict

    Args:
        d ([type]): the corresponding dict error_vectors
        f ([type]): function to map over tensor

    Returns:
        [type]: f(tensor)
    """
    if d is None:
        return None
    return dict(map(lambda item: (item[0], f(item[1])), [x for x in d.items() if x[1].shape[0] != 0]))



class HybridZonotope(AbstractElement):
    def __init__(self, head: Tensor, beta: Optional[Tensor], errors: Optional[Union[Dict[str,Tensor],Tensor]], domain: str, error_id:Optional[str]=None, ch_zono_errors: Optional[List[str]] = None) -> None:
        super(HybridZonotope,self).__init__()
        self.head = head
        self.beta = beta
        self.domain = domain
        assert self.domain in ['box', 'hbox', 'zono', 'chzono']
        self.device = self.head.device
        self.reset_errors(errors, error_id)
        self.ch_zono_errors = ch_zono_errors
        assert not torch.isnan(self.head).any()

    def add_box_error(self, error:Tensor) -> None:
        if self.beta is None:
            self.beta = error
        else:
            self.beta += error.abs()

    def add_zono_error(self, errors:Tensor, error_id:Optional[str]=None, chzono_err:bool=False) -> None:
        if (errors != 0.).any() or error_id is not None or len(self.zono_errors) == 0:
            error_id = str(uuid.uuid4()) if error_id is None else error_id
            self.zono_errors[error_id] = errors
            if chzono_err:
                assert self.domain == 'chzono'
                if self.ch_zono_errors is None:
                    self.ch_zono_errors = [error_id]
                else:
                    self.ch_zono_errors.append(error_id)

    def reset_errors(self, errors:Optional[Union[Dict[str,Tensor],Tensor]], error_id:Optional[str]=None) -> None:
        self.zono_errors = {}
        self.ch_zono_errors = []
        if isinstance(errors, torch.Tensor):
            self.add_zono_error(errors, error_id)
        elif isinstance(errors, dict):
            self.zono_errors = errors
        elif errors is not None:
            assert False, f"Errors with type {type(errors)} are not supported."

    def get_errors(self, include_ch_box_term=False) -> Tensor:
        if self.zono_errors is None:
            return None
        keys = set(self.zono_errors.keys())
        if self.domain == 'chzono' and not include_ch_box_term and self.ch_zono_errors is not None:
            keys = keys - set(self.ch_zono_errors)
        return torch.cat([self.zono_errors[k] for k in keys], dim=0)#.detach().squeeze()

    @classmethod
    def construct_from_noise(cls, x: Tensor, eps: Union[float,Tensor], domain: str, dtype: torch.dtype=torch.float32,
                             data_range: Tuple[float,float]=(0, 1)) -> "HybridZonotope":
        # compute center and error terms from input, perturbation size and data range
        assert data_range[0] < data_range[1]
        x_center, x_beta = clamp_image(x, eps, data_range[0], data_range[1])
        x_center, x_beta = x_center.to(dtype=dtype), x_beta.to(dtype=dtype)
        return cls.construct(x_center, x_beta, domain=domain)

    @classmethod
    def construct_from_bounds(cls, min_x: Tensor, max_x: Tensor, dtype: torch.dtype=torch.float32,
                              domain: str="box") -> "HybridZonotope":
        # compute center and error terms from elementwise bounds
        assert min_x.shape == max_x.shape
        x_center, x_beta = head_from_bounds(min_x,max_x)
        x_center, x_beta = x_center.to(dtype=dtype), x_beta.to(dtype=dtype)
        return cls.construct(x_center, x_beta, domain=domain)

    @staticmethod
    def construct(x_center: Tensor, x_beta: Tensor, domain: str="box") -> "HybridZonotope":
        device = x_center.device
        if domain == 'box':
            return HybridZonotope(x_center, x_beta, None, domain)
        elif domain in ['zono', 'zono_iter', 'hbox', 'chzono']:
            batch_size = x_center.size()[0]
            n_elements = x_center[0].numel()
            # construct error coefficient matrix
            # ei = torch.eye(n_elements).expand(batch_size, n_elements, n_elements).permute(1, 0, 2).to(device)
            # if len(x_center.size()) > 2:
            #     ei = ei.contiguous().view(n_elements, *x_center.size())
            ei = HybridZonotope.get_error_matrix(x_center)
            # update beta tensor to account for errors captures by error coefficients
            new_beta = None if "zono" in domain else torch.zeros(x_beta.shape).to(device=device, dtype=torch.float32)

            return HybridZonotope(x_center, new_beta, ei * x_beta.unsqueeze(0), domain)
        else:
            raise RuntimeError('Unsupported HybridZonotope domain: {}'.format(domain))

    @staticmethod
    def get_error_matrix(x, error_idx=None):
        batch_size = x.size()[0]
        if error_idx is None:
            n_elements_e = x[0].numel()
            ei = torch.eye(n_elements_e, dtype=x.dtype, device=x.device).expand(batch_size, n_elements_e, n_elements_e).permute(1, 0, 2)
        else:
            assert batch_size == 1
            n_elements_e = int(error_idx.sum())
            n_elements_x = x[0].numel()
            ei = torch.zeros((n_elements_e, n_elements_x), dtype=x.dtype, device=x.device)
            ei[torch.arange(n_elements_e).view(-1,1), error_idx.flatten().nonzero()] = 1
            ei = ei.expand(batch_size, n_elements_e, n_elements_x).permute(1, 0, 2)
        if len(x.size()) > 2:
            ei = ei.contiguous().view(n_elements_e, *x.size())
        return ei

    @staticmethod
    def get_new_errs(approx_indicator: torch.Tensor, x_center_new: torch.Tensor,
                     x_beta_new: torch.Tensor) -> torch.Tensor:
        device = x_center_new.device
        dtype = x_center_new.dtype
        batch_size, center_shape = x_center_new.size()[0], x_center_new.size()[1:]

        # accumulate error position over batch dimension
        new_err_pos = (approx_indicator.long().sum(dim=0) > 0).nonzero()
        num_new_errs = new_err_pos.size()[0]
        err_idx_dict = {tuple(pos.cpu().numpy()): idx for pos, idx in zip(new_err_pos,range(num_new_errs))}

        nnz = approx_indicator.nonzero()
        # extract error sizes
        beta_values = x_beta_new[tuple(nnz[:, i] for i in range((nnz.shape[1])))]
        # generate new error matrix portion
        new_errs = torch.zeros((num_new_errs, batch_size,) + center_shape).to(device, dtype=dtype)
        new_errs[([err_idx_dict[tuple(key[1:].cpu().numpy())] for key in nnz],) + tuple(nnz[:, i] for i in range((nnz.shape[1])))]= beta_values
        return new_errs

    def dim(self):
        return self.head.dim()

    # @staticmethod
    # def join(x: List["HybridZonotope"], trunk_errors: Union[None,Tensor,int]=None, dim: Union[Tensor,int]=0,
    #          mode: str="cat") -> "HybridZonotope":
    #     # x is list of HybridZonotopesk
    #     # trunk_errors is number of last shared error between all Hybrid zonotopes, usually either number of initial
    #     # errors or number of errors at point where a split between network branches occured
    #     device = x[0].head.device
    #     if mode not in ["cat","stack"]:
    #         raise RuntimeError(f"Unkown join mode : {mode:}")

    #     if mode == "cat":
    #         new_head = torch.cat([x_i.head for x_i in x], dim=dim)
    #     elif mode == "stack":
    #         new_head = torch.stack([x_i.head for x_i in x], dim=dim)

    #     if all([x_i.beta is None for x_i in x]):
    #         new_beta = None
    #     elif any([x_i.beta is None for x_i in x]):
    #         assert False, "Mixed HybridZonotopes can't be joined"
    #     else:
    #         if mode == "cat":
    #             new_beta = torch.cat([x_i.beta for x_i in x], dim=dim)
    #         elif mode == "stack":
    #             new_beta = torch.stack([x_i.beta for x_i in x], dim=dim)

    #     if all([x_i.errors is None for x_i in x]):
    #         new_errors = None
    #     elif any([x_i.errors is None for x_i in x]):
    #         assert False, "Mixed HybridZonotopes can't be joined"
    #     else:
    #         if trunk_errors is None:
    #             trunk_errors = [0 for x_i in x]
    #         exit_errors = [x_i.errors.size()[0]-trunk_errors[i] for i, x_i in enumerate(x)] # number of additional errors for every Hybrid zonotope
    #         tmp_errors= [None for _ in x]
    #         for i, x_i in enumerate(x):
    #             tmp_errors[i] = torch.cat([x_i.errors[:trunk_errors[i]],
    #                                torch.zeros([max(trunk_errors) - trunk_errors[i] + sum(exit_errors[:i])]
    #                                            + list(x_i.errors.size()[1:])).to(device),
    #                                x_i.errors[trunk_errors[i]:],
    #                                torch.zeros([sum(exit_errors[i + 1:])] + list(x_i.errors.size()[1:])).to(device)],
    #                                dim=0)

    #         if mode == "cat":
    #             new_errors = torch.cat(tmp_errors, dim=dim + 1)
    #         elif mode == "stack":
    #             new_errors = torch.stack(tmp_errors, dim=dim+1)

    #     return HybridZonotope(new_head,
    #                           new_beta,
    #                           new_errors,
    #                           x[0].domain)

    def size(self, idx=None):
        if idx is None:
            return self.head.size()
        else:
            return self.head.size(idx)

    def view(self, size):
        return HybridZonotope(self.head.view(*size),
                              None if self.beta is None else self.beta.view(size),
                              map_dict(self.zono_errors, lambda x: x.view(x.size()[0], *size)),
                              self.domain)

    def flatten(self):
        bsize = self.head.size(0)
        return self.view((bsize, -1))


    def normalize(self, mean: Tensor, sigma: Tensor) -> "HybridZonotope":
        return (self - mean) / sigma

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self) -> "HybridZonotope":
        new_head = -self.head
        new_beta = None if self.beta is None else self.beta
        new_errors = map_dict(self.zono_errors, lambda x: -x)
        return HybridZonotope(new_head, new_beta, new_errors, self.domain, ch_zono_errors=self.ch_zono_errors)

    def __add__(self, other: Union[Tensor, float, int, "HybridZonotope"]) -> "HybridZonotope":
        if isinstance(other, torch.Tensor) or isinstance(other, float) or isinstance(other, int):
            return HybridZonotope(self.head + other, self.beta, self.zono_errors, self.domain, ch_zono_errors=self.ch_zono_errors)
        elif isinstance(other, HybridZonotope):
            #assert self.domain == other.domain
            return self.add(other)
        else:
            assert False, 'Unknown type of other object'

    def __truediv__(self, other: Union[Tensor, int, float]) -> "HybridZonotope":
        if isinstance(other, torch.Tensor) or isinstance(other, float) or isinstance(other, int) or isinstance(other, torch.Tensor):
            return HybridZonotope(self.head / other,
                                  None if self.beta is None else self.beta / abs(other),
                                  map_dict(self.zono_errors, lambda x: x/other),
                                  self.domain, ch_zono_errors=self.ch_zono_errors)
        else:
            assert False, 'Unknown type of other object'

    def __mul__(self, other: Union[Tensor, int, float]) -> "HybridZonotope":
        #doesnt work for tensor I guess
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, int) or (isinstance(other, torch.Tensor)):
            d = self.head.device
            return HybridZonotope((self.head * other).to(d),
                                    None if self.beta is None else (self.beta * abs(other)).to(d),
                                    None if self.zono_errors is None else map_dict(self.zono_errors, lambda x: (x * other).to(d)),
                                    self.domain, ch_zono_errors=self.ch_zono_errors)
        elif isinstance(other, HybridZonotope):
            return self.prod(other)
        else:
            assert False, 'Unknown type of other object'

    def __rmul__(self, other):  # Assumes associativity
        return self.__mul__(other)

    def __getitem__(self, indices) -> "HybridZonotope":
        if not isinstance(indices, tuple):
            indices = tuple([indices])
        return HybridZonotope(self.head[indices],
                              None if self.beta is None else self.beta[indices],
                              map_dict(self.zono_errors, lambda x: x[(slice(None), *indices)]),
                              self.domain, ch_zono_errors=self.ch_zono_errors)

    def clone(self) -> "HybridZonotope":
        return HybridZonotope(self.head.clone(),
                              None if self.beta is None else self.beta.clone(),
                              map_dict(self.zono_errors, lambda x: x.clone()),
                              self.domain, ch_zono_errors = None if self.ch_zono_errors is None else self.ch_zono_errors.copy())

    def detach(self) -> "HybridZonotope":
        return HybridZonotope(self.head.detach(),
                              None if self.beta is None else self.beta.detach(),
                              map_dict(self.zono_errors, lambda x: x.detach()),
                              self.domain,  ch_zono_errors = None if self.ch_zono_errors is None else self.ch_zono_errors.copy())

    def abs(self) -> "HybridZonotope":
        assert 'zono' not in self.domain
        lb, ub = self.concretize()
        is_cross = (lb < 0) & (ub > 0)
        ub_out = torch.maximum(ub.abs(), lb.abs())
        lb_out = torch.where(is_cross, torch.zeros_like(lb), torch.minimum(ub.abs(), lb.abs()))
        center = (lb_out + ub_out) / 2
        beta = (ub_out - lb_out) / 2
        return HybridZonotope(center, beta, None, self.domain, ch_zono_errors=None if self.ch_zono_errors is None else self.ch_zono_errors.copy())

    def pad(self, padding, mode):
        new_head = F.pad(self.head, padding, mode=mode)
        new_beta = None if self.beta is None else F.pad(self.beta.view(-1, *self.head.shape[1:]), padding, mode=mode)
        new_errors = map_dict(self.zono_errors, lambda x: F.pad(x.view(-1, *self.head.shape[1:]), padding, mode=mode).view(-1, *new_head.shape))
        return HybridZonotope(new_head, new_beta, new_errors, self.domain, ch_zono_errors=None if self.ch_zono_errors is None else self.ch_zono_errors.copy())

    def fft_conv(self, w_fft, transpose=False):
        new_head = fft_conv(self.head, w_fft, transpose)
        new_beta = None if self.beta is None else fft_conv(self.beta.view(-1, *self.head.shape[1:]), w_fft, transpose)
        new_errors = map_dict(self.zono_errors, lambda x: fft_conv(x.view(-1, *self.head.shape[1:]), w_fft, transpose).view(-1, *new_head.shape))
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def conv_transpose2d(self, weight, bias, stride, padding, output_padding, groups, dilation):
        new_head = F.conv_transpose2d(self.head, weight, bias, stride, padding, output_padding, groups, dilation)
        new_beta = None if self.beta is None else F.conv_transpose2d(self.beta.view(-1, *self.head.shape[1:]), weight.abs(), None, stride, padding, output_padding, groups, dilation)
        new_errors = map_dict(self.zono_errors, lambda x: F.conv_transpose2d(x.view(-1, *self.head.shape[1:]), weight, bias, stride, padding, output_padding, groups, dilation).view(-1, *new_head.shape))
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def border_relu(self, deepz_lambda, bounds, init_lambda=False, border=None):
        assert border is not None
        def set_boarder_zero(x, border):
            x[:, :, :border, :] = 0
            x[:, :, -border:, :] = 0
            x[:, :, :, :border] = 0
            x[:, :, :, -border:] = 0
            return x

        new_zono, _ = self.relu(deepz_lambda, bounds, init_lambda)

        new_head = set_boarder_zero(new_zono.head, border)
        new_beta = None if new_zono.beta is None else set_boarder_zero(new_zono.beta.view(-1, *new_zono.head.shape[1:]), border)
        new_errors = map_dict(new_zono.zono_errors, lambda x: set_boarder_zero(x.view(-1, *new_zono.head.shape[1:]), border).view(-1, *new_head.shape))
        return HybridZonotope(new_head, new_beta, new_errors, new_zono.domain, ch_zono_errors=None if new_zono.ch_zono_errors is None else new_zono.ch_zono_errors.copy())

    def reciprocal(self) -> "HybridZonotope":
        assert 'zono' not in self.domain
        lb, ub = self.concretize()
        lbr, ubr = 1.0 / lb, 1.0 / ub
        is_cross = (lb < 0) & (ub > 0)
        lb_out = torch.where(is_cross, -np.inf * torch.ones_like(ubr), ubr)
        ub_out = torch.where(is_cross, np.inf * torch.ones_like(ubr), lbr)
        center = (lb_out + ub_out) / 2
        beta = (ub_out - lb_out) / 2
        return HybridZonotope(center, beta, None, self.domain, ch_zono_errors=self.ch_zono_errors)

    def max_center(self) -> Tensor:
        return self.head.max(dim=1)[0].unsqueeze(1)

    def avg_pool2d(self, kernel_size: int, stride:int) -> "HybridZonotope":
        new_head = F.avg_pool2d(self.head, kernel_size, stride)
        new_beta = None if self.beta is None else F.avg_pool2d(self.beta.view(-1, *self.head.shape[1:]), kernel_size, stride)
        new_errors = map_dict(self.zono_errors, lambda x: F.avg_pool2d(x.view(-1, *self.head.shape[1:]), kernel_size, stride).view(-1, *new_head.shape))
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def widen(self, mul:float, add:float):
        new_head = self.head
        new_beta = add * torch.ones_like(new_head) if self.beta is None else add + self.beta * mul
        new_errors = map_dict(self.zono_errors, lambda x: x*mul)
        return HybridZonotope(new_head, new_beta, new_errors, self.domain, ch_zono_errors = None if self.ch_zono_errors is None else self.ch_zono_errors.copy())

    def global_avg_pool2d(self) -> "HybridZonotope":
        new_head = F.adaptive_avg_pool2d(self.head, 1)
        new_beta = None if self.beta is None else F.adaptive_avg_pool2d(self.beta.view(-1, *self.head.shape[1:]), 1)
        new_errors = map_dict(self.zono_errors, lambda x: F.adaptive_avg_pool2d(x.view(-1, *self.head.shape[1:]), 1).view(-1, *new_head.shape))
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def max_pool2d(self, kernel_size, stride):
        if self.zono_errors is not None:
            assert False, "MaxPool for Zono not Implemented"
        lb, ub = self.concretize()
        new_lb = F.max_pool2d(lb, kernel_size, stride)
        new_ub = F.max_pool2d(ub, kernel_size, stride)
        return HybridZonotope.construct_from_bounds(new_lb, new_ub, self.dtype ,self.domain)

    def conv2d(self, weight:Tensor, bias:Tensor, stride:int, padding:int, dilation:int, groups:int) -> "HybridZonotope":
        new_head = F.conv2d(self.head, weight, bias, stride, padding, dilation, groups)
        new_beta = None if self.beta is None else F.conv2d(self.beta, weight.abs(), None, stride, padding, dilation, groups)
        new_errors = map_dict(self.zono_errors, lambda x: x.view(-1, *x.size()[2:]))
        new_errors = map_dict(new_errors, lambda x:  F.conv2d(x, weight, None, stride, padding, dilation, groups) )
        if new_errors is not None:
            for key, val in new_errors.items():
                new_errors[key] = val.view(self.zono_errors[key].size()[0], self.zono_errors[key].size()[1], *new_head.shape[1:])
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def convtranspose2d(self, weight:Tensor, bias:Tensor, stride:int, padding:int,  output_padding:int, dilation:int, groups:int) -> "HybridZonotope":
        new_head = F.conv_transpose2d(self.head, weight, bias, stride, padding, output_padding,  dilation, groups)
        new_beta = None if self.beta is None else F.conv_transpose2d(self.beta, weight.abs(), None, stride, padding, output_padding,  dilation, groups)
        new_errors = map_dict(self.zono_errors, lambda x: x.view(-1, *x.size()[2:]))
        new_errors = map_dict(new_errors, lambda x: F.conv_transpose2d(x, weight, None, stride, padding, output_padding, dilation, groups))
        if new_errors is not None:
            for key, val in new_errors.items():
                new_errors[key] = val.view(self.zono_errors[key].size()[0], self.zono_errors[key].size()[1], *new_head.shape[1:])
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def can_contain(self, other):
        lb, ub = self.concretize()
        other_lb, other_ub = other.concretize()
        contained = (lb <= other_lb) & (other_ub <= ub)
        cont_factor = 2 * torch.max(((other_ub - self.head) / (ub - lb + 1e-16)).abs().max(),
                                    ((other_lb - self.head) / (ub - lb + 1e-16)).abs().max())
        return contained.all(), cont_factor

    def contains(self, other: "HybridZonotope", method='basis', verbose: Optional[bool]=False):
        assert(self.head.size(0) == 1)
        start = time()

        #lp method based on Theorem 3 in Sadraddini et al. (2018) in gurobi
        if method == 'gurobi':
            this_zono = HybridZonotope(self.head, None, self.zono_errors, domain=self.domain, ch_zono_errors=self.ch_zono_errors)
            if self.beta is not None:
                ei = HybridZonotope.get_error_matrix(self.head)
                this_zono.add_zono_error(ei * self.beta.unsqueeze(0), chzono_err=True)
            # if hasattr(self,"ch_zono_errors"):
            #     this_zono.ch_zono_errors = self.ch_zono_errors

            other_zono = HybridZonotope(other.head, None, other.zono_errors, domain=other.domain, ch_zono_errors=other.ch_zono_errors)
            if other.beta is not None:
                ei = HybridZonotope.get_error_matrix(other.head)
                other_zono.add_zono_error(ei * other.beta.unsqueeze(0), chzono_err=True)
            # if hasattr(other_zono,"ch_zono_errors"):
            #     other_zono.ch_zono_errors = other.ch_zono_errors
            assert this_zono.beta is None
            assert other_zono.beta is None
            model = Model("Zonotope Containment")
            model.setParam("OutputFlag", 0)
            eps = 1e-6
            model.setParam(GRB.Param.FeasibilityTol, eps)
            X = other_zono.get_errors(include_ch_box_term=True)
            nx = X.size(0)
            X = X.view(nx, -1).T
            x = other_zono.head
            Y = this_zono.get_errors(include_ch_box_term=True)
            ny = Y.size(0)
            Y = Y.view(ny, -1).T
            y = this_zono.head
            diff = (y - x).view(-1)
            n = diff.size(0)

            # variables
            beta = [model.addVar(vtype=GRB.CONTINUOUS, lb=-1, ub=1, name=f"beta_{i}") for i in range(ny)]
            Gamma = [[0 for _ in range(nx)] for _ in range(ny)]
            Lambda1 = [[0 for _ in range(nx)] for _ in range(ny)]
            Lambda2 = [[0 for _ in range(nx)] for _ in range(ny)]
            Lambda3 = [[0 for _ in range(nx)] for _ in range(ny)]
            Lambda4 = [[0 for _ in range(nx)] for _ in range(ny)]
            for i in range(ny):
                for j in range(nx):
                    Gamma[i][j] = model.addVar(vtype=GRB.CONTINUOUS, lb=-1, ub=1, name=f"Gamma_{i}_{j}")
                    Lambda1[i][j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Lambda1_{i}_{j}")
                    Lambda2[i][j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Lambda2_{i}_{j}")
                    Lambda3[i][j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Lambda3_{i}_{j}")
                    Lambda4[i][j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Lambda4_{i}_{j}")
                    model.addConstr(Lambda1[i][j] - Lambda2[i][j] == Gamma[i][j])
                    model.addConstr(Lambda4[i][j] - Lambda3[i][j] == Gamma[i][j])

            for i in range(n):
                # beta
                row = []
                for j in range(ny):
                    row.append(Y[i, j].item() * beta[j])
                model.addConstr(quicksum(row) == diff[i].item())

                # Gamma 
                for j in range(nx):
                    pos = []
                    for k in range(ny):
                        pos.append(Y[i, k].item() * Gamma[k][j])
                    model.addConstr(quicksum(pos) == X[i, j].item())

            for i in range(ny):
                gi = quicksum(Lambda1[i] + Lambda2[i])
                model.addConstr(gi - beta[i] <= 1)
                gi = quicksum(Lambda3[i] + Lambda4[i])
                model.addConstr(gi + beta[i] <= 1)

            #obj = quicksum(model.getVars())
            #model.setObjective(obj, GRB.MINIMIZE)
            model.update()
            model.reset()
            model.optimize()
            if model.Status != 2:
                print(f"GUROBI contained: False, model runtime: {model.runtime:.4f}, total time: {time() - start:.4f}")
                return False, np.inf
            else:
                print(f"GUROBI contained: True, model runtime: {model.runtime:.4f}, total time: {time() - start:.4f}")
                for i in range(ny):
                    beta[i] = beta[i].X
                beta = torch.tensor(beta, dtype=self.dtype)
                for i in range(ny):
                    for j in range(nx):
                        Gamma[i][j] = Gamma[i][j].X
                Gamma = torch.tensor(Gamma, dtype=self.dtype)
                # assert (Y@beta-diff).abs().max().item() <= eps
                # assert ((Y@Gamma)-X).abs().max().item() <= eps
                #assert (Gamma.abs().sum(1) + beta.abs() <= 1+2*eps).all()
                return True, 1-eps



        # 'basis' method
        elif method.startswith("basis"):
            if 'zono' not in self.domain and 'zono' not in other.domain: #interval
                lb, ub = self.concretize()
                other_lb, other_ub = other.concretize()
                contained = (lb <= other_lb) & (other_ub <= ub)
                cont_factor = 2*torch.max(((other_ub-self.head)/(ub-lb+1e-16)).abs().max(), ((other_lb-self.head)/(ub-lb+1e-16)).abs().max())
                return contained.all(), cont_factor
            else:
                dtype = self.head.dtype
                device = self.head.device

                # Solve the LGS that we get when representing the "other" zonotope in the "self" basis

                # System Ax = B
                # NOTE: This gives us eps parameterized vectors in x space (i.e. shape 40x824)

                #A should be consolidated, i.e. we would have only one error term in dicts
                A = self.get_errors(include_ch_box_term=False).flatten(start_dim=1).T  # containing init errors
                B = other.get_errors(include_ch_box_term=False).flatten(start_dim=1).T  #contained init errors

                if not hasattr(self, "errors_inv"):
                    self.errors_inv = None

                if A.shape[-1] == A.shape[-2] and self.errors_inv is None:
                    try:
                        self.errors_inv = torch.inverse(A)
                    except:
                        if verbose:
                            print(f"Failed to inverse error matrix.")
                        pass
                        # return False, np.inf

                if self.errors_inv is None:
                    if A.shape[0] != A.shape[1]:
                        # sol = np.linalg.lstsq(A.cpu().detach().numpy(), B.cpu().detach().numpy(), rcond=None)
                        # x = torch.tensor(sol[0], dtype=dtype, device=device)
                        x = torch.lstsq(B, A)[0][:A.shape[1]]
                    elif float(torch.__version__[:-2].split("+")[0][:4].rstrip(".")) < 1.9:
                        try:
                            x = torch.solve(B, A).solution
                        except Exception as e:
                            print("Encountered exception:")
                            print(e)
                            return False, np.inf
                    else:
                        x = torch.linalg.solve(A, B)
                else:
                    x = torch.matmul(self.errors_inv, B)

                # Note sometimes we dont have full rank for A (check sol[1]) - the solution however has no residuals
                # Here x contains the coordinates of the inner zonotope in the outer basis
                if not torch.isclose(torch.matmul(A, x), B, atol=1e-7,rtol=1e-6).all():  # "Projection of contained errors into base of containing errors failed"
                    uncaptured_errors = torch.abs(B - torch.matmul(A, x)).sum(1)
                    if verbose:
                        print(f"Inverse led to uncaptured errors.")
                else:
                    uncaptured_errors = torch.zeros_like(self.head).flatten()

                # Sum the absolutes row-wise to get the scaling factor for the containing error coefficients to overapproximated the contained ones
                abs_per_orig_vector = x.abs().sum(1)
                max_sp = torch.max(abs_per_orig_vector).cpu().item()
                if str(max_sp) == "nan":
                    if verbose:
                        print(f"Containment of zono errors failed with {max_sp}")
                    return False, max_sp

                if method == "basis_new":
                    uncaptured_errors += (torch.maximum(abs_per_orig_vector-1, torch.zeros_like(abs_per_orig_vector)).unsqueeze(0)*A).abs().sum(1)
                    abs_per_orig_vector = torch.minimum(abs_per_orig_vector, torch.ones_like(abs_per_orig_vector))
                    if verbose:
                        print(f"Using method {method}")
                elif max_sp > 1:
                    if verbose:
                        print(f"Containment of zono errors failed with {max_sp}")
                    return False, max_sp

                # Here we adjust for the head displacement
                diff = torch.abs(self.head - other.head).detach().view(-1)

                # Here we adjust for errors not captured by the intial matching due to differences in spanned space:
                diff += uncaptured_errors.view(-1)

                # Here we just do a basic check on the "independent" merge errors
                self_beta = self.concretize('box', chzono_as_box=True)[1].detach()
                # Ensure that all beta of the outer zono actually induce a box
                if self.ch_zono_errors is not None and not all([((self.zono_errors[x] != 0).sum(0) <= 1).all() for x in self.ch_zono_errors]):
                    if verbose:
                        print("Some box terms not captured by zono errors")
                    return False, np.inf

                other_beta = other.concretize('box', chzono_as_box=True)[1].detach()
                merge_diff = (self_beta - other_beta).view(-1)
                # When the merge errors of the containing zono are larger than that of the contained one, we can use this extra to compensate for some of the difference in the heads
                diff = torch.maximum(diff - merge_diff, torch.zeros_like(diff)).detach()

                # This projects the remaining difference between the heads into the error coefficient matrix
                diff = torch.diag(diff.view(-1))
                if self.errors_inv is None:
                    if A.shape[0] != A.shape[1]:
                        sol_diff = np.linalg.lstsq(A.cpu().numpy(), diff.cpu().numpy(), rcond=None)
                        x_diff = torch.tensor(sol_diff[0], dtype=dtype, device=device)
                    elif float(torch.__version__[:-2]) < 1.9:
                        x_diff = torch.solve(diff, A).solution
                    else:
                        x_diff = torch.linalg.solve(A, diff)
                else:
                    x_diff = torch.matmul(self.errors_inv, diff)

                if not torch.isclose(torch.matmul(A, x_diff), diff, atol=1e-7, rtol=1e-6).all():
                    # f"Projection of head difference into base of containing errors failed"
                    if verbose:
                        print("Some box terms not captured by zono errors")
                    return False, np.inf

                abs_per_orig_vector_diff = abs_per_orig_vector + x_diff.abs().sum(1)
                max_sp_diff = torch.max(abs_per_orig_vector_diff).cpu().item()

                # Check if with this additional component, we are still contained
                if max_sp_diff > 1 or str(max_sp_diff) == "nan":
                    if verbose:
                        print(f"Containment of box terms failed with {max_sp_diff}")
                    return False, max_sp_diff

                if verbose:
                    print(f"Containment with {max_sp_diff}")

                return True, max_sp_diff
        else:
            assert False, f"containment method {method} unknown"

    @property
    def shape(self):
        return self.head.shape

    @property
    def dtype(self):
        return self.head.dtype

    def dim(self):
        return self.head.dim()

    def linear(self, weight:Tensor, bias:Union[Tensor,None], C:Union[Tensor,None]=None) -> "HybridZonotope":
        if C is None:
            if bias is None:
                return self.matmul(weight.t())
            else:
                return self.matmul(weight.t()) + bias.unsqueeze(0)
        else:
            if bias is None:
                return self.unsqueeze(-1).rev_matmul(C.matmul(weight)).squeeze()
            else:
                return self.unsqueeze(-1).rev_matmul(C.matmul(weight)).squeeze() + C.matmul(bias)

    def matmul(self, other: Tensor) -> "HybridZonotope":
        return HybridZonotope(self.head.matmul(other),
                              None if self.beta is None else self.beta.matmul(other.abs()),
                              map_dict(self.zono_errors, lambda x: x.matmul(other)),
                              self.domain)

    def rev_matmul(self, other: Tensor) -> "HybridZonotope":
        return HybridZonotope(other.matmul(self.head),
                              None if self.beta is None else other.abs().matmul(self.beta),
                              map_dict(self.zono_errors, lambda x: other.matmul(x)),
                              self.domain)

    def fft(self):
        assert self.beta is None
        return HybridZonotope(torch.fft.fft2(self.head).real,
                         None,
                         map_dict(self.zono_errors, lambda x: torch.fft.fft2(x).real),
                         self.domain)

    def batch_norm(self, bn) -> "HybridZonotope":
        view_dim_list = [1, -1] + (self.head.dim()-2)*[1]
        self_stat_dim_list = [0, 2, 3] if self.head.dim()==4 else [0]
        if bn.training:
            momentum = 1 if bn.momentum is None else bn.momentum
            mean = self.head.mean(dim=self_stat_dim_list).detach()
            var = self.head.var(unbiased=False, dim=self_stat_dim_list).detach()
            if bn.running_mean is not None and bn.running_var is not None and bn.track_running_stats:
                bn.running_mean = bn.running_mean * (1 - momentum) + mean * momentum
                bn.running_var = bn.running_var * (1 - momentum) + var * momentum
            else:
                bn.running_mean = mean
                bn.running_var = var
        c = (bn.weight / torch.sqrt(bn.running_var + bn.eps))
        b = (-bn.running_mean*c + bn.bias)
        new_head = self.head*c.view(*view_dim_list)+b.view(*view_dim_list)
        new_errors = map_dict(self.zono_errors, lambda x: x*c.view(*([1]+view_dim_list)))
        new_beta = None if self.beta is None else self.beta * c.abs().view(*view_dim_list)
        return HybridZonotope(new_head, new_beta, new_errors, self.domain, ch_zono_errors=self.ch_zono_errors)

    # def batch_norm(self, bn, mean, var):
    #     view_dim_list = [1, -1]+(self.head.dim()-2)*[1]
    #     assert mean is not None and var is not None
    #     c = (bn.weight / torch.sqrt(var + bn.eps))
    #     b = (-mean*c + bn.bias)
    #     new_head = self.head*c.view(*view_dim_list)+b.view(*view_dim_list)
    #     new_errors = None if self.errors is None else self.errors * c.view(*([1]+view_dim_list))
    #     new_beta = None if self.beta is None else self.beta * c.abs().view(*view_dim_list)
    #     return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    @staticmethod
    def cat(zonos, dim=0):
        new_head = torch.cat([x.head for x in zonos], dim)
        new_beta = None if all([x.beta is None for x in zonos]) else \
            torch.cat([x.beta if x.beta is not None else torch.zeros_like(x.head) for x in zonos], dim)
        dtype = zonos[0].head.dtype
        device = zonos[0].head.device

        zono_error_keys = []
        for zono in zonos:
            zono_error_keys += list(zono.zono_errors.keys()) if zono.zono_errors is not None else []
        zono_error_keys = list(set(zono_error_keys))

        new_errors = {}
        for key in zono_error_keys:
            n_err = [x.zono_errors[key].shape[0] for x in zonos if key in x.zono_errors.keys()][0]
            new_error = torch.cat([x.zono_errors[key] if key in x.zono_errors.keys() else
                                   torch.zeros((n_err, *x.shape), dtype=dtype, device=device) for x in zonos], dim+1)
            new_errors[key] = new_error
        new_ch_zono_errors = []
        for zono in zonos:
            if zono.ch_zono_errors is not None:
                new_ch_zono_errors += zono.ch_zono_errors

        return HybridZonotope(new_head, new_beta, new_errors, zonos[0].domain, ch_zono_errors=list(set(new_ch_zono_errors)))

    def relu(self, deepz_lambda:Union[None,Tensor]=None, bounds:Union[None,Tensor]=None, init_lambda=False) -> Tuple["HybridZonotope",Union[Tensor,None]]:
        lb, ub = self.concretize()
        D = 1e-6

        if self.domain == "box":
            min_relu, max_relu = F.relu(lb), F.relu(ub)
            return HybridZonotope(0.5 * (max_relu + min_relu), 0.5 * (max_relu - min_relu), None, self.domain), None
        elif self.domain == "hbox":
            is_under = (ub <= 0)
            is_above = (ub > 0) & (lb >= 0)
            is_cross = (ub > 0) & (lb < 0)

            new_head = self.head.clone()
            new_beta = self.beta.clone()
            assert len(self.zono_errors) == 1
            key, val = list(self.zono_errors.items())[0]
            new_errors = val.clone()

            ub_half = ub / 2

            new_head[is_under] = 0
            new_head[is_cross] = ub_half[is_cross]

            new_beta[is_under] = 0
            new_beta[is_cross] = ub_half[is_cross]

            new_errors[:, ~is_above] = 0
            new_errors = {key: new_errors}

            return HybridZonotope(new_head, new_beta, new_errors, self.domain), None
        elif "zono" in self.domain:
            if bounds is not None:
                lb_refined, ub_refined = bounds
                lb = torch.max(lb_refined, lb)
                ub = torch.min(ub_refined, ub)

            is_cross = (lb < 0) & (ub > 0)
            dtype = self.dtype
            device = self.device

            relu_lambda = torch.where(is_cross, ub/(ub-lb+D), (lb >= 0).float().to(dtype=self.dtype))
            if deepz_lambda is not None:
                if (deepz_lambda < 0).all():
                    deepz_lambda.data = relu_lambda.detach()
                relu_mu = torch.where(deepz_lambda < relu_lambda, 0.5 * ub * (1 - deepz_lambda),
                                      -0.5 * lb * deepz_lambda)
                relu_lambda = torch.clamp(deepz_lambda, 0, 1)
                relu_mu = torch.where(is_cross.detach(), relu_mu, torch.zeros(lb.size(), dtype=dtype, device=device))
                relu_lambda = torch.where(is_cross.detach(), relu_lambda, (lb >= 0).to(dtype))
            else:
                relu_mu = torch.where(is_cross, -0.5*ub*lb/(ub-lb+D), torch.zeros(lb.size(), dtype=self.dtype).to(self.device))
                deepz_lambda = None

            assert (not torch.isnan(relu_mu).any()) and (not torch.isnan(relu_lambda).any())

            new_head = self.head * relu_lambda + relu_mu
            assert (not torch.isnan(relu_mu).any())
            assert (not torch.isnan(new_head).any()) 
            out = HybridZonotope(new_head, None, map_dict(self.zono_errors, lambda x: x * relu_lambda), self.domain)
            new_errs = self.get_new_errs(is_cross, new_head, relu_mu)
            if self.domain == 'chzono':
                # make chzono box terms into zonotope terms 
                self.ch_zono_errors = []
                out.add_zono_error(new_errs, chzono_err=True)
            else:
                out.add_zono_error(new_errs)

            return out, deepz_lambda
        else:
            raise RuntimeError("Error applying ReLU with unkown domain: {}".format(self.domain))

    def log(self, deepz_lambda:Union[None,Tensor]=None, bounds:Union[None,Tensor]=None) -> Tuple["HybridZonotope",Union[Tensor,None]]:
        lb, ub = self.concretize()
        D = 1e-6

        if bounds is not None:
            lb_refined, ub_refined = bounds
            lb = torch.max(lb_refined, lb)
            ub = torch.min(ub_refined, ub)

        assert (lb >= 0).all()

        if self.domain in ["box", "hbox"]:
            min_log, max_log = lb.log(), ub.log()
            return HybridZonotope(0.5 * (max_log + min_log), 0.5 * (max_log - min_log), None, "box"), None
        assert self.beta is None

        is_tight = (ub == lb)

        log_lambda_s = (ub.log() - (lb+D).log()) / (ub - lb + D)
        if self.domain == 'zono_iter' and deepz_lambda is not None:
            if not ((deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()):
                deepz_lambda = ((log_lambda_s - (1/ub)) / (1/(lb+D)-1/ub)).detach().requires_grad_(True)
            log_lambda = deepz_lambda * (1 / (lb+D)) + (1 - deepz_lambda) * (1 / ub)
        else:
            log_lambda = log_lambda_s

        log_lb_0 = torch.where(log_lambda < log_lambda_s, ub.log() - ub * log_lambda, lb.log() - lb * log_lambda)
        log_mu = 0.5 * (-log_lambda.log() - 1 + log_lb_0)
        log_delta = 0.5 * (-log_lambda.log() - 1 - log_lb_0)

        log_lambda = torch.where(is_tight, torch.zeros_like(log_lambda), log_lambda)
        log_mu = torch.where(is_tight, ub.log(), log_mu)
        log_delta = torch.where(is_tight, torch.zeros_like(log_delta), log_delta)

        assert (not torch.isnan(log_mu).any()) and (not torch.isnan(log_lambda).any())

        new_head = self.head * log_lambda + log_mu
        out = HybridZonotope(new_head, None, map_dict(self.zono_errors, lambda x: x * log_lambda), self.domain)
        new_errs = self.get_new_errs(~is_tight, new_head, log_delta)
        out.add_zono_error(new_errs)
        assert (not torch.isnan(new_head).any()) and (not torch.isnan(new_errs).any())
        return out, deepz_lambda

    def exp(self, deepz_lambda:Union[None,Tensor]=None, bounds:Union[None,Tensor]=None) -> Tuple["HybridZonotope",Union[Tensor,None]]:
        lb, ub = self.concretize()
        D = 1e-6

        if bounds is not None:
            lb_refined, ub_refined = bounds
            lb = torch.max(lb_refined, lb)
            ub = torch.min(ub_refined, ub)

        if self.domain in ["box", "hbox"]:
            min_exp, max_exp = lb.exp(), ub.exp()
            return HybridZonotope(0.5 * (max_exp + min_exp), 0.5 * (max_exp - min_exp), None, "box"), None

        assert self.beta is None

        is_tight = (ub.exp() - lb.exp()).abs() < 1e-15

        exp_lambda_s = (ub.exp() - lb.exp()) / (ub - lb + D)
        if self.domain == 'zono_iter' and deepz_lambda is not None:
            if not ((deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()):
                deepz_lambda = ((exp_lambda_s -lb.exp()) / (ub.exp()-lb.exp()+D)).detach().requires_grad_(True)
            exp_lambda = deepz_lambda * torch.min(ub.exp(), (lb + 1 - 0.05).exp()) + (1 - deepz_lambda) * lb.exp()
        else:
            exp_lambda = exp_lambda_s
        exp_lambda = torch.min(exp_lambda, (lb + 1 - 0.05).exp()) # ensure non-negative output only

        for _ in range(2): #First go with minimum area (while non negative lb)/ provided slopes. If this leads to negative/zero lower bounds, fall back to minimum slope
            exp_ub_0 = torch.where(exp_lambda > exp_lambda_s, lb.exp() - exp_lambda * lb, ub.exp() - exp_lambda * ub)
            exp_mu = 0.5 * (exp_lambda * (1 - exp_lambda.log()) + exp_ub_0)
            exp_delta = 0.5 * (exp_ub_0 - exp_lambda * (1 - exp_lambda.log()))

            exp_lambda = torch.where(is_tight, torch.zeros_like(exp_lambda), exp_lambda)
            exp_mu = torch.where(is_tight, ub.exp(), exp_mu)
            exp_delta = torch.where(is_tight, torch.zeros_like(exp_delta), exp_delta)

            assert (not torch.isnan(exp_mu).any()) and (not torch.isnan(exp_lambda).any())

            new_head = self.head * exp_lambda + exp_mu
            new_zono = HybridZonotope(new_head, None, map_dict(self.zono_errors, lambda x: x * exp_lambda), self.domain)
            new_errs = self.get_new_errs(~is_tight, new_head, exp_delta)
            new_zono.add_zono_error(new_errs)
            assert (not torch.isnan(new_head).any()) and (not torch.isnan(new_errs).any())
            new_zono = HybridZonotope(new_head, None, new_errs, self.domain)
            if (new_zono.concretize()[0] <= 0).any():
                exp_lambda = torch.where(new_zono.concretize()[0] <= 0, lb.exp(), exp_lambda) #torch.zeros_like(exp_lambda)
            else:
                break

        return HybridZonotope(new_head, None, new_zono.zono_errors, self.domain), deepz_lambda

    # def inv(self, deepz_lambda, bounds):
    #     lb, ub = self.concretize()
    #     assert (lb > 0).all()
    #
    #     if self.domain in ["box", "hbox"]:
    #         min_inv, max_inv = 1 / ub, 1 / lb
    #         return HybridZonotope(0.5 * (max_inv + min_inv), 0.5 * (max_inv - min_inv), None, "box"), None
    #
    #     assert self.beta is None
    #
    #     if bounds is not None:
    #         lb_refined, ub_refined = bounds
    #         lb = torch.max(lb_refined, lb)
    #         ub = torch.min(ub_refined, ub)
    #
    #     assert (lb > 0).all()
    #
    #     is_tight = (ub == lb)
    #
    #     inv_lambda_s = -1 / (ub * lb)
    #     if self.domain == 'zono_iter' and deepz_lambda is not None:
    #         # assert (deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()
    #         if not ((deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()):
    #             deepz_lambda = (-ub*lb+lb**2)/(lb**2-ub**2)
    #         inv_lambda = deepz_lambda * (-1 / lb**2) + (1 - deepz_lambda) * (-1 / ub**2)
    #     else:
    #         inv_lambda = inv_lambda_s
    #
    #
    #     inv_ub_0 = torch.where(inv_lambda > inv_lambda_s, 1 / lb - inv_lambda * lb, 1 / ub - inv_lambda * ub)
    #     inv_mu = 0.5 * (2 * (-inv_lambda).sqrt() + inv_ub_0)
    #     inv_delta = 0.5 * (inv_ub_0 - 2 * (-inv_lambda).sqrt())
    #
    #     # inv_mu = torch.where(inv_lambda > inv_lambda_s, 0.5 * (2 * (-inv_lambda).sqrt() + 1 / lb - inv_lambda * lb),
    #     #                      0.5 * (2 * (-inv_lambda).sqrt() + 1 / ub - inv_lambda * ub))
    #
    #     inv_lambda = torch.where(is_tight, torch.zeros_like(inv_lambda), inv_lambda)
    #     inv_mu = torch.where(is_tight, 1/ub, inv_mu)
    #     inv_delta = torch.where(is_tight, torch.zeros_like(inv_delta), inv_delta)
    #
    #     assert (not torch.isnan(inv_mu).any()) and (not torch.isnan(inv_lambda).any())
    #
    #     new_head = self.head * inv_lambda + inv_mu
    #     old_errs = self.errors * inv_lambda
    #     new_errs = self.get_new_errs(~is_tight, new_head, inv_delta)
    #     new_errors = torch.cat([old_errs, new_errs], dim=0)
    #     assert (not torch.isnan(new_head).any()) and (not torch.isnan(new_errors).any())
    #     return HybridZonotope(new_head, None, new_errors, self.domain), deepz_lambda

    def sum(self, dim:int, reduce_dim=False) -> "HybridZonotope":
        new_head = self.head.sum(dim=dim)
        new_beta = None if self.beta is None else self.beta.abs().sum(dim=dim)
        new_errors = map_dict(self.zono_errors, lambda x: x.sum(dim=dim))

        if not reduce_dim:
            new_head = new_head.unsqueeze(dim)
            new_beta = None if new_beta is None else new_beta.unsqueeze(dim)
            new_errors = map_dict(new_errors, lambda x: x.unsqueeze(dim+1))

        assert not torch.isnan(new_head).any()
        assert new_beta is None or not torch.isnan(new_beta).any()
        assert new_errors is None or not any([torch.isnan(v).any() for v in new_errors.values()])
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def unsqueeze(self, dim:int) -> "HybridZonotope":
        new_head = self.head.unsqueeze(dim)
        new_beta = None if self.beta is None else self.beta.unsqueeze(dim)
        new_errors = map_dict(self.zono_errors, lambda x: x.unsqueeze(dim+1))

        assert not torch.isnan(new_head).any()
        assert new_beta is None or not torch.isnan(new_beta).any()
        assert new_errors is None or not any([torch.isnan(v).any() for v in new_errors.values()])
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def squeeze(self, dim:Union[None,int]=None) -> "HybridZonotope":
        if dim is None:
            new_head = self.head.squeeze()
            new_beta = None if self.beta is None else self.beta.squeeze()
            new_errors = map_dict(self.zono_errors, lambda x: x.squeeze())
        else:
            new_head = self.head.squeeze(dim)
            new_beta = None if self.beta is None else self.beta.squeeze(dim)
            new_errors = map_dict(self.zono_errors, lambda x: x.squeeze(dim+1))

        assert not torch.isnan(new_head).any()
        assert new_beta is None or not torch.isnan(new_beta).any()
        assert new_errors is None or not any([torch.isnan(v).any() for v in new_errors.values()])
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)


    def add(self, summand_zono:"HybridZonotope", shared_errors:int=0) -> "HybridZonotope":
        assert all([x == y or x == 1 or y == 1 for x, y in zip(self.head.shape[::-1], summand_zono.head.shape[::-1])])
        dtype = self.head.dtype
        device = self.head.device

        new_head = self.head + summand_zono.head
        if self.beta is None and summand_zono.beta is None:
            new_beta = None
        elif self.beta is not None and summand_zono.beta is not None:
            new_beta = self.beta.abs() + summand_zono.beta.abs()
        else:
            new_beta = self.beta if self.beta is not None else summand_zono.beta

        if self.zono_errors == None:
            new_dict = summand_zono.zono_errors
        elif summand_zono.zono_errors == None:
            new_dict = self.zono_errors
        else:
            new_dict = {}
            for key1, val1 in self.zono_errors.items():
                val2 = summand_zono.zono_errors.get(key1)
                if val2 is not None:
                    new_dict[key1] = val1 + val2
                else:
                    new_dict[key1] = val1

            for key2, val2 in summand_zono.zono_errors.items():
                contained = new_dict.get(key2)
                if contained is None:
                    new_dict[key2] = val2

        assert not torch.isnan(new_head).any()
        assert new_beta is None or not torch.isnan(new_beta).any()
        
        #new_domain = summand_zono.domain if new_beta is None else ("hbox" if new_dict is not None else "box")
        new_domain = self.domain
        return HybridZonotope(new_head, new_beta, new_dict, new_domain)

    def prod(self, factor_zono:"HybridZonotope", low_mem:bool=False) -> "HybridZonotope":
        dtype = self.head.dtype
        device = self.head.device
        lb_self, ub_self = self.concretize()
        lb_other, ub_other = factor_zono.concretize()

        if self.domain == factor_zono.domain:
            domain = self.domain
        elif "box" in [self.domain, factor_zono.domain]:
            domain = "box"
        elif "hbox" in [self.domain, factor_zono.domain]:
            domain = "hbox"
        else:
            assert False

        if domain in ["box", "hbox"] or low_mem:
            min_prod = torch.min(torch.min(torch.min(lb_self*lb_other, lb_self*ub_other), ub_self*lb_other), ub_self*ub_other)
            max_prod = torch.max(torch.max(torch.max(lb_self*lb_other, lb_self*ub_other), ub_self*lb_other), ub_self*ub_other)
            return HybridZonotope(0.5 * (max_prod + min_prod), 0.5 * (max_prod - min_prod), None, "box")
        assert self.beta is None

        assert all([x==y or x==1 or y ==1 for x,y in zip(self.head.shape[::-1],factor_zono.head.shape[::-1])])
        self_errors = set(self.zono_errors.keys())
        other_errors = set(factor_zono.zono_errors.keys())
        shared_errors = self_errors.intersection(other_errors)

        new_head = self.head * factor_zono.head
        new_errors = {} 
        for key in shared_errors:
            err_o = factor_zono.zono_errors[key] 
            err_s = self.zono_errors[key] 
            new_errors[key] = self.head.unsqueeze(0) * err_o + factor_zono.head.unsqueeze(0) * err_s

        for key in self_errors - shared_errors:
            new_errors[key] = factor_zono.head.unsqueeze(0) * self.zono_errors[key] 
        for key in other_errors - shared_errors:
            new_errors[key] = self.head.unsqueeze(0) * factor_zono.zono_errors[key] 

        quad_errs = torch.zeros_like(new_head)
        for key_s, key_o in product(self_errors, other_errors):
            err_s = self.zono_errors[key_s]
            err_o = factor_zono.zono_errors[key_o]
            if err_s.size(0) == 0 or err_o.size(0) == 0: break
            err_s = (err_s * torch.ones((err_s.size(0),) + tuple(factor_zono.head.shape), dtype=dtype, device=device))
            err_o = (err_o * torch.ones((err_o.size(0),) + tuple(self.head.shape), dtype=dtype, device=device))
            diag_errs = (err_s * err_o).abs().sum(dim=0)
            quad_term = err_s.unsqueeze(0) * err_o.unsqueeze(1)
            quad_term = (quad_term + quad_term.transpose(0, 1)).abs().sum(dim=1).sum(dim=0) - diag_errs
            quad_errs += 0.5 * quad_term
            new_head += 0.5 * diag_errs

        assert (not torch.isnan(new_head).any()) and all([not torch.isnan(err).any() for err in new_errors.values()])
        new_zono = HybridZonotope(new_head, None, new_errors, self.domain)

        new_zono.add_zono_error(self.get_new_errs(torch.ones(self.head.shape), new_head, quad_errs))
        return new_zono

    def join(self, other:"HybridZonotope") -> "HybridZonotope":
        # join from Taylor1+
        dtype = self.head.dtype
        device = self.head.device
        lb_self, ub_self = self.concretize()
        lb_other, ub_other = other.concretize()

        lb = torch.minimum(lb_self, lb_other)
        ub = torch.maximum(ub_self, ub_other)
        new_head = 0.5 * (lb + ub)

        if self.domain == other.domain:
            domain = self.domain
        elif "box" in [self.domain, other.domain]:
            domain = "box"
        elif "hbox" in [self.domain, other.domain]:
            domain = "hbox"
        else:
            assert False
        if domain in ["box", "hbox"]:
            return HybridZonotope(new_head, 0.5 * (ub - lb), None, domain)

        assert self.head.shape == other.head.shape
        self_errors = set(self.zono_errors.keys())
        other_errors = set(other.zono_errors.keys())
        shared_errors = self_errors.intersection(other_errors)

        new_errors = {} 
        for key in shared_errors:
            err_o = other.zono_errors[key] 
            err_s = self.zono_errors[key] 
            err_lb = torch.minimum(err_s, err_o)
            err_ub = torch.maximum(err_s, err_o)
            cross = (err_lb < 0) & (err_ub >= 0)
            new_error_term = torch.where(cross, torch.zeros_like(err_lb), torch.minimum(err_lb.abs(), err_ub.abs()))
            if (new_error_term>0.).any():
                new_errors[key] = new_error_term

        # zero coefficient for all non-shared errors

        new_zono = HybridZonotope(new_head, None, new_errors, self.domain)
        new_err_term = torch.maximum(ub - new_zono.concretize()[1], new_zono.concretize()[0] -lb) + 1e-10
        new_zono.add_zono_error(self.get_new_errs(torch.ones(self.head.shape), new_head, new_err_term))
        assert (new_zono.concretize()[0]<= lb).all() and (new_zono.concretize()[1] >= ub).all()
        return new_zono

    def consolidate_errors(self, new_basis_vectors=None, basis_transform_method="pca", verbosity = 0, error_eps_add=1e-8, error_eps_mul=1e-8):
        if new_basis_vectors is None:
            errors_to_get_basis_from = self.get_errors()
            shp = errors_to_get_basis_from.shape
            errors_to_get_basis_from = errors_to_get_basis_from.view(shp[0], -1)
            new_basis_vectors = self.get_new_basis(errors_to_get_basis_from, basis_transform_method)
            new_basis_vectors = torch.tensor(new_basis_vectors, dtype=self.dtype, device=self.device)
        return self.move_to_basis(new_basis_vectors, error_eps_add, error_eps_mul), new_basis_vectors

    def move_to_basis(self, new_basis, ERROR_EPS_ADD=0, ERROR_EPS_MUL=0):
        if self.zono_errors is None:
            return HybridZonotope(self.head, None, self.box_errors, self.domain)
        else:
            new_zono, _ = self.basis_transform(new_basis, ERROR_EPS_ADD, ERROR_EPS_MUL)
            return new_zono

    def upsample(self, size:int, mode:str, align_corners:bool, consolidate_errors:bool=True) -> "HybridZonotope":
        new_head = F.interpolate(self.head, size=size, mode=mode, align_corners=align_corners)
        delta = 0
        assert mode in ["nearest","linear","bilinear","trilinear"], f"Upsample"

        if self.beta is not None:
            new_beta = F.interpolate(self.beta, size=size, mode=mode, align_corners=align_corners)
            delta = delta + new_beta
        else:
            new_beta = None

        if self.errors is not None:
            errors_resized = self.errors.view(-1, *self.head.shape[1:])
            new_errors = F.interpolate(errors_resized, size=size, mode=mode, align_corners=align_corners)
            new_errors = new_errors.view(-1, *new_head.shape)
            delta = delta + new_errors.abs().sum(0)
        else:
            new_errors = None

        if consolidate_errors:
            return HybridZonotope.construct_from_bounds(new_head-delta, new_head+delta, domain=self.domain)
        else:
            return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def beta_to_error(self):
        out = HybridZonotope(self.head, None, self.zono_errors, self.domain)
        if self.beta is None:
            return out
        out.add_zono_error(self.beta)
        return out

    def concretize(self, mode='all', chzono_as_box=False) -> Tuple[Tensor,Tensor]:
        delta = 0
        if self.zono_errors is not None and mode in ["all", "zono"]:
            keys = set(self.zono_errors.keys())
            if self.domain == 'chzono' and chzono_as_box and self.ch_zono_errors is not None:
                keys = keys - set(self.ch_zono_errors)
            for key in keys:
                val = self.zono_errors[key]
                delta = delta + val.abs().sum(0)

        if mode in ["all", "box"]:
            if self.beta is not None: 
                delta = delta + self.beta.abs().sum(0)

            if self.domain == 'chzono' and chzono_as_box and self.ch_zono_errors is not None:
                for key in self.ch_zono_errors:
                    val = self.zono_errors[key]
                    delta = delta + val.abs().sum(0)

        return self.head * (mode == "all") - delta, self.head * (mode == "all") + delta

    def avg_width(self) -> Tensor:
        lb, ub = self.concretize()
        return (ub - lb).mean()

    def is_greater(self, i:int, j:int, threshold_min:Union[Tensor,float]=0) -> Tuple[Tensor,bool]:
        diff_head = self.head[:, i] - self.head[:, j]
        delta = diff_head
        if self.zono_errors is not None:
            diff_errors = None
            for val in self.zono_errors.values():
                de = (val[..., i] - val[..., j]).abs().sum(dim=0)
                if diff_errors is None:
                    diff_errors = de
                else:
                    diff_errors += de 
            if diff_errors is not None:
                delta -= diff_errors
        if self.beta is not None:
            diff_beta = (self.beta[:, i] + self.beta[:, j]).abs()
            delta -= diff_beta
        return delta, delta > threshold_min

    def get_min_diff(self, i, j):
        """ returns minimum of logit[i] - logit[j] """
        return self.is_greater(i, j)[0]

    def verify(self, targets: Tensor, threshold_min:Union[Tensor,float]=0, corr_only:bool=False) -> Tuple[Tensor,Tensor,Tensor]:
        n_class = self.head.size()[1]
        verified = torch.zeros(targets.size(), dtype=torch.uint8).to(self.head.device)
        verified_corr = torch.zeros(targets.size(), dtype=torch.uint8).to(self.head.device)
        if n_class == 1:
            # assert len(targets) == 1
            verified_list = torch.cat([self.concretize()[1] < threshold_min, self.concretize()[0] >= threshold_min], dim=1)
            verified[:] = torch.any(verified_list, dim=1)
            verified_corr[:] = verified_list.gather(dim=1,index=targets.long().unsqueeze(dim=1)).squeeze(1)
            threshold = torch.cat(self.concretize(),1).gather(dim=1, index=(1-targets).long().unsqueeze(dim=1)).squeeze(1)
        else:
            threshold = np.inf * torch.ones(targets.size(), dtype=torch.float).to(self.head.device)
            for i in range(n_class):
                if corr_only and i not in targets:
                    continue
                isg = torch.ones(targets.size(), dtype=torch.uint8).to(self.head.device)
                margin = np.inf * torch.ones(targets.size(), dtype=torch.float).to(self.head.device)
                for j in range(n_class):
                    if i != j and isg.any():
                        margin_tmp, ok = self.is_greater(i, j, threshold_min)
                        margin = torch.min(margin, margin_tmp)
                        isg = isg & ok.byte()
                verified = verified | isg
                verified_corr = verified_corr | (targets.eq(i).byte() & isg)
                threshold = torch.where(targets.eq(i).byte(), margin, threshold)
        return verified, verified_corr, threshold

    def get_wc_logits(self, targets:Tensor, use_margins:bool=False)->Tensor:
        n_class = self.size(-1)
        device = self.head.device

        if use_margins:
            def get_c_mat(n_class, target):
                return torch.eye(n_class, dtype=torch.float32)[target].unsqueeze(dim=0) \
                       - torch.eye(n_class, dtype=torch.float32)
            if n_class > 1:
                c = torch.stack([get_c_mat(n_class,x) for x in targets], dim=0)
                self = -(self.unsqueeze(dim=1) * c.to(device)).sum(dim=2, reduce_dim=True)
        batch_size = targets.size()[0]
        lb, ub = self.concretize()
        if n_class == 1:
            wc_logits = torch.cat([ub, lb],dim=1)
            wc_logits = wc_logits.gather(dim=1, index=targets.long().unsqueeze(1))
        else:
            wc_logits = ub.clone()
            wc_logits[np.arange(batch_size), targets] = lb[np.arange(batch_size), targets]
        return wc_logits

    def ce_loss(self, targets:Tensor) -> Tensor:
        wc_logits = self.get_wc_logits(targets)
        if wc_logits.size(1) == 1:
            return F.binary_cross_entropy_with_logits(wc_logits.squeeze(1), targets.float(), reduction="none")
        else:
            return F.cross_entropy(wc_logits, targets.long(), reduction="none")

    def to(self, device):
        return HybridZonotope(self.head.to(device),
                          None if self.beta is None else self.beta.to(device),
                          None if self.zono_errors is None else {k:v.to(device) for k,v in self.zono_errors.items()},
                          self.domain)

    def basis_transform(self, new_basis: Tensor, ERROR_EPS_ADD: Optional[float]=0., ERROR_EPS_MUL:Optional[float]=0.) -> Tuple["HybridZonotope", Tensor]:
        # We solve for the coordinates (x) of curr_basis (B) in new_basis (A)
        # I.e. we solve Ax=b
        assert self.shape[0] == 1
        A = new_basis
        B = self.get_errors()
        shp = B.shape
        B = B.view(shp[0], -1).T

        device = self.device
        dtype = self.dtype

        if A.shape[0] < 500 or A.shape[0] != A.shape[1]:
            # depending on the size of the matrices different methods are faster
            # TODO look at this again
            if isinstance(A, torch.Tensor):
                A = A.cpu().detach().numpy()
            B = B.cpu().detach().numpy()
            sol = np.linalg.lstsq(A, B, rcond=None)[0]
            assert np.isclose(np.matmul(A, sol), B, atol=1e-7, rtol=1e-6).all()
            sol = torch.tensor(sol, dtype=dtype, device=device)
        else:
            if not isinstance(A, torch.Tensor):
                A = torch.tensor(A)
            # if not A.device.type == "cpu":
            #     A = A.cpu()
            sol = torch.solve(B, A).solution

            assert torch.isclose(torch.matmul(A, sol), B, atol=1e-7, rtol=1e-6).all(), f"Projection into new base errors failed"

        # We add the component ERROR_EPS_ADD to ensure the resulting error matrix has full rank and to compensate for potential numerical errors
        x = torch.sum(sol.abs(), dim=1) * (1 + ERROR_EPS_MUL) + ERROR_EPS_ADD


        out = HybridZonotope(self.head, self.beta, None, self.domain)
        new_errors = (x.reshape(1, -1) * new_basis).T.unsqueeze(1).view(-1, *self.head.shape)
        out.add_zono_error(new_errors)
        if self.domain == 'chzono' and self.ch_zono_errors is not None:
            for k in self.ch_zono_errors:
               out.add_zono_error(self.zono_errors[k], k, True)
        return out, x


    # def get_zono_matrix(self):
    #     if self.zono_errors is None:
    #         return None
    #     return torch.cat(list(self.zono_errors.values()), dim=0)

    def concretize_into_basis(self, basis):
        shp = self.head.shape
        all_as_errors = self.beta_to_error()
        delta = all_as_errors.basis_transform(basis)[1]

        if isinstance(basis, torch.Tensor):
            A = basis.cpu().detach().numpy()
        B = torch.flatten(self.head, start_dim=1).cpu().detach().numpy().T
        sol = np.linalg.lstsq(A, B, rcond=None)[0].T

        new_head = torch.tensor(sol, dtype=self.dtype, device=self.device)

        return (new_head - delta).view(shp), (new_head + delta).view(shp)

    def concretize_into_box(self):
        lb, ub = self.concretize()
        return HybridZonotope.construct_from_bounds(lb, ub, domain='box')

    def to_zonotope(self):
        out = HybridZonotope(self.head, None, self.zono_errors, domain='zono')
        if self.beta is not None:
            out.add_zono_error(self.get_new_errs(torch.ones(self.head.shape), self.head, self.beta))
        return out

    def get_new_basis(self, errors_to_get_basis_from: Optional[Tensor]=None, method:Optional[str]="pca"):
        """
        Compute a bais of error directions from errors_to_get_basis_from
        :param errors_to_get_basis_from: Error matrix to be overapproximated
        :param method: "pca" or "pca_zero_mean"
        :return: a basis of error directions
        """

        if errors_to_get_basis_from is None:
            errors_to_get_basis_from = self.errors

        if method == "pca":
            U, S, Vt = np.linalg.svd((errors_to_get_basis_from - errors_to_get_basis_from.mean(0)).cpu(),
                                     full_matrices=False)
            max_abs_cols = np.argmax(np.abs(U), axis=0)
            signs = np.sign(U[max_abs_cols, range(U.shape[1])])
            Vt *= signs[:, np.newaxis]
            new_basis_vectors = Vt.T
            ### Torch version is much (factor 6) slower despite avoiding move of data to cpu
            # U, S, V = torch.svd(errors_to_get_basis_from - errors_to_get_basis_from.mean(0), some=True)
            # max_abs_cols = U.abs().argmax(0)
            # signs = U[max_abs_cols, range(U.shape[1])].sign()
            # new_basis_vectors_2 = V*signs.unsqueeze(0)
        elif method == "pca_zero_mean":
            U, S, Vt = np.linalg.svd(errors_to_get_basis_from.cpu(), full_matrices=False)
            max_abs_cols = np.argmax(np.abs(U), axis=0)
            signs = np.sign(U[max_abs_cols, range(U.shape[1])])
            Vt *= signs[:, np.newaxis]
            new_basis_vectors = Vt.T
        else: assert False, f"Unknown method {method}"

        return new_basis_vectors

