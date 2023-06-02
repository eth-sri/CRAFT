"""
Based on HybridZonotope from DfifAI (https://github.com/eth-sri/diffai/blob/master/ai.py)
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from torch import Tensor

from ai.ai_util import AbstractElement, head_from_bounds


def apply_block(x, block):
    curr = x 

    #('upsample', nn.Upsample(scale_factor=2**level_diff, mode='nearest'))
    for l in block: 
        if isinstance(l, torch.nn.modules.conv.Conv2d):
            curr = curr.conv2d(l.weight, l.bias, l.stride, l.padding, l.dilation, l.groups)
        elif isinstance(l, torch.nn.modules.batchnorm.BatchNorm2d):
            curr = curr.normalize(l.running_mean.view(1, -1, 1, 1),
                                    torch.sqrt(l.running_var+l.eps).view(1, -1, 1, 1))
        elif isinstance(l, torch.nn.modules.activation.ReLU):
            curr = curr.relu(None, None, None)
        else:
            print(type(l))
            assert(False)
    return curr


def get_patch_masks(x, label, pos='random', n_random=3, p_size=2):
    #input x shape:     batch_size, chan, width, height
    #output mask shape:     batch_size, patches, chan=1, width, height
    #input label shape:     batch_size,
    #output label shape:     batch_size, patches,

    assert pos in ['all', 'random']
    shape = x.shape
    assert(len(shape) == 4)
    batch_size, chan, width, height = shape
    if n_random == -1:
        pos = 'all'

    w_max_patches = width - p_size + 1
    h_max_patches = height - p_size + 1
    assert(w_max_patches >= 1)
    assert(h_max_patches >= 1)

    if pos == 'all':
        n_patches = w_max_patches * h_max_patches
        startw = torch.arange(w_max_patches).repeat(h_max_patches)
        starth = torch.repeat_interleave(torch.arange(h_max_patches), w_max_patches)
    else:
        n_patches = n_random
        startw = torch.randint(low=0, high=w_max_patches, size=(batch_size * n_patches, ))
        starth = torch.randint(low=0, high=h_max_patches, size=(batch_size * n_patches, ))
    w, h = [], []
    for i in range(p_size):
        for j in range(p_size):
            w.append(startw + i)
            h.append(starth + j)
    w, h = torch.stack(w).t().reshape(-1), torch.stack(h).t().reshape(-1)
    if pos == 'all':
        w, h = w.repeat(batch_size), h.repeat(batch_size)

    batch_idx = torch.arange(batch_size).repeat_interleave(n_patches)
    patch_idx = torch.arange(n_patches).repeat(batch_size)

    s = p_size * p_size
    batch_idx = batch_idx.repeat_interleave(s)
    patch_idx = patch_idx.repeat_interleave(s)

    idx = torch.stack([batch_idx, patch_idx, torch.zeros((batch_idx.size(0),), dtype=torch.long), w, h], dim=-1)

    mask = torch.sparse.FloatTensor(idx.t(), torch.ones((idx.size(0),)),
                                    torch.Size([batch_size, n_patches, 1, width, height])).to(x.device)
    return mask


def clamp_patch(x, eps, mask):
    mask = mask.to_dense()
    mask.unsqueeze_(1)
    chan = x.shape[1]
    d = torch.repeat_interleave(eps * mask, chan, dim=1)
    min_x = torch.clamp(x - d, min=0)
    max_x = torch.clamp(x + d, max=1)
    x_center = 0.5 * (max_x + min_x)
    x_beta = 0.5 * (max_x - min_x)
    return x_center, x_beta


def clamp_image(x, eps, data_lb=0, data_ub=1):
    if isinstance(eps, torch.Tensor):
        assert(len(eps.size()) == 1 and eps.size(0) == x.size(0))
        s = [1] * len(x.size())
        s[0] = eps.size(0)
        eps = eps.view(s)
    min_x = torch.clamp(x-eps, min=data_lb)
    max_x = torch.clamp(x+eps, max=data_ub)
    x_center = 0.5 * (max_x + min_x)
    x_beta = 0.5 * (max_x - min_x)
    return x_center, x_beta


def get_new_errs(should_box, newhead, newbeta):
    new_err_pos = (should_box.long().sum(dim=0) > 0).nonzero()
    num_new_errs = new_err_pos.size()[0]
    nnz = should_box.nonzero()
    if len(newhead.size()) == 2:
        batch_size, n = newhead.size()[0], newhead.size()[1]
        ids_mat = torch.zeros(n, dtype=torch.long).to(newhead.device)
        ids_mat[new_err_pos[:, 0]] = torch.arange(num_new_errs).to(newhead.device)
        beta_values = newbeta[nnz[:, 0], nnz[:, 1]]
        new_errs = torch.zeros((num_new_errs, batch_size, n)).to(newhead.device, dtype=newhead.dtype)
        err_ids = ids_mat[nnz[:, 1]]
        new_errs[err_ids, nnz[:, 0], nnz[:, 1]] = beta_values
    else:
        batch_size, n_channels, img_dim = newhead.size()[0], newhead.size()[1], newhead.size()[2]
        ids_mat = torch.zeros((n_channels, img_dim, img_dim), dtype=torch.long).to(newhead.device)
        ids_mat[new_err_pos[:, 0], new_err_pos[:, 1], new_err_pos[:, 2]] = torch.arange(num_new_errs).to(newhead.device)
        beta_values = newbeta[nnz[:, 0], nnz[:, 1], nnz[:, 2], nnz[:, 3]]
        new_errs = torch.zeros((num_new_errs, batch_size, n_channels, img_dim, img_dim)).to(newhead.device, dtype=newhead.dtype)
        err_ids = ids_mat[nnz[:, 1], nnz[:, 2], nnz[:, 3]]
        new_errs[err_ids, nnz[:, 0], nnz[:, 1], nnz[:, 2], nnz[:, 3]] = beta_values
    return new_errs


class HybridZonotope(AbstractElement):

    def __init__(self, head, beta, errors, domain):
        self.head = head
        self.beta = beta
        self.errors = errors
        self.domain = domain
        self.device = self.head.device
        assert not torch.isnan(self.head).any()
        assert self.beta is None or (not torch.isnan(self.beta).any())
        assert self.errors is None or (not torch.isnan(self.errors).any())
        assert not (self.errors is not None and self.beta is not None) # currently do either zonotope or box

    # @staticmethod
    # def zonotope_from_noise(x, eps, domain='zono', dtype=torch.float32):
    #     if domain == "box":
    #         return HybridZonotope.box_from_noise(x, eps)
    #     batch_size = x.size()[0]
    #     n_elements = x[0].numel()
    #     ei = torch.eye(n_elements).expand(batch_size, n_elements, n_elements).permute(1, 0, 2).to(x.device)
    #     x_center, x_beta = clamp_image(x, eps)
    #     x_center, x_beta = x_center.to(dtype=dtype), x_beta.to(dtype=dtype)
    #     if len(x.size()) > 2:
    #         ei = ei.contiguous().view(n_elements, *x.size())
    #     return HybridZonotope(x_center, None, ei * x_beta.unsqueeze(0), domain)

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
        elif domain in ['zono', 'zono_iter', 'hbox']:
            batch_size = x_center.size()[0]
            n_elements = x_center[0].numel()
            # construct error coefficient matrix
            ei = torch.eye(n_elements).expand(batch_size, n_elements, n_elements).permute(1, 0, 2).to(device)
            if len(x_center.size()) > 2:
                ei = ei.contiguous().view(n_elements, *x_center.size())

            # update beta tensor to account for errors captures by error coefficients
            new_beta = None if "zono" in domain else torch.zeros(x_beta.shape).to(device=device, dtype=torch.float32)

            return HybridZonotope(x_center, new_beta, ei * x_beta.unsqueeze(0), domain)
        else:
            raise RuntimeError('Unsupported HybridZonotope domain: {}'.format(domain))

    @staticmethod
    def zonotope_patch_from_mask(x, mask, domain='zono', dtype=torch.float32):
        b_size, chan, h, w = x.size()
        assert(mask.size(0) == b_size)
        n_patches = mask.size(1)
        assert(mask.size(2) == 1)
        assert(mask.size(3) == h)
        assert(mask.size(4) == w)

        x_center = x.repeat_interleave(n_patches, dim=0).to(dtype=dtype).contiguous()
        #mask = # b_size * n_patches, h, w
        dmask = mask.to_dense().to(dtype=dtype).repeat_interleave(chan, dim=2)
        x_center[dmask.view(-1, chan, h, w) > 0] = 0.5
        del dmask

        i = mask._indices() #assumes indices were not changed after creation
        n_el = i.size(1) // (b_size * n_patches)
        d = torch.arange(n_el, device=i.device).repeat_interleave(b_size*n_patches).view((1, -1))
        c = torch.arange(chan, device=i.device).repeat_interleave(i.size(1)).view((1, -1))
        i = torch.cat([d, i], dim=0).repeat(1, chan)
        i[3, :] = c
        errs = torch.sparse.FloatTensor(i, 0.5 * torch.ones((i.size(1),), device=i.device),
                                        torch.Size([n_el, b_size, n_patches, chan, w, h])).to_dense().view(n_el, -1, chan, w, h)

        return HybridZonotope(x_center, None, errs, domain)

    
    @staticmethod
    def zonotope_patch_from_noise(x, eps, label, domain, pos='random', n_random=10, p_size=2, dtype=torch.float32):
        shape = x.shape
        x_center, x_beta, l = clamp_patch(x, eps, label, pos=pos, n_random=n_random, p_size=p_size)
        assert(len(shape) == 4)
        batch_size, chan, width, height = shape
        n_elements = x[0].numel()
        ei = torch.eye(n_elements).expand(x_center.shape[0], n_elements, n_elements).permute(1, 0, 2).to(x.device)
        ei = ei.contiguous().view(n_elements, *x_center.size())
        errs = ei * x_beta.unsqueeze(0) # could be made sparse
        return HybridZonotope(x_center, None, errs, domain), l
    
    @staticmethod
    def box_from_noise(x, eps):
        x_center, x_beta = clamp_image(x, eps)
        return HybridZonotope(x_center, x_beta, None, 'box')

    def size(self):
        return self.head.size()

    def zeros(self, size):
        assert self.head.size(0) == size[0]
        zero = torch.zeros(size, device=self.head.device, dtype=self.head.dtype)
        if self.errors is not None: #zonotpe
            nerr = self.errors.size(0)
            s = [nerr] + list(size)
            return HybridZonotope(zero,
                             None,
                             torch.zeros(s, device=self.head.device, dtype=self.head.dtype),
                             self.domain)
        else: #Interval
            return HybridZonotope(zero,
                             zero,
                             None,
                             self.domain)


    def view(self, size):
        return HybridZonotope(self.head.view(*size),
                              None if self.beta is None else self.beta.view(size),
                              None if self.errors is None else self.errors.view(self.errors.size()[0], *size),
                              self.domain)

    def flatten(self):
        bsize = self.head.size(0)
        return self.view((bsize, -1))


    def normalize(self, mean, sigma):
        return (self - mean) / sigma

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self) -> "HybridZonotope":
        new_head = -self.head
        new_beta = None if self.beta is None else self.beta
        new_errors = None if self.errors is None else -self.errors
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def __add__(self, other):
        if isinstance(other, torch.Tensor):
            return HybridZonotope(self.head + other, self.beta, self.errors, self.domain)
        elif isinstance(other, HybridZonotope):
            if self.errors is None and other.errors is None:
                if self.beta is None:
                    if other.beta is None:
                        new_beta = None
                    else:
                        new_beta = other.beta
                else:
                    if other.beta is None:
                        new_beta = self.beta
                    else:
                        new_beta = self.beta + other.beta
                return HybridZonotope(self.head + other.head, new_beta, None, self.domain)
            else:
                #this assumes both have the same eps -- hope you know what you are doing
                errs1 = self.errors
                errs2 = other.errors
                if errs1.size(0) > errs2.size(0):
                    errs = errs1.clone()
                    print(errs1.size(), errs2.size(), errs[:errs2.size(0), ...].size())
                    errs[:errs2.size(0), ...] = errs[:errs2.size(0), ...] + errs2
                else:
                    errs = errs2.clone()
                    print(errs1.size(), errs2.size(), errs[:errs1.size(0), ...].size())
                    errs[:errs1.size(0), ...] = errs[:errs1.size(0), ...] + errs1
                return HybridZonotope(self.head + other.head, None, errs, self.domain)
        else:
            assert False, 'Unknown type of other object'

    def __truediv__(self, other: Union[Tensor, int, float]) -> "HybridZonotope":
        if isinstance(other, torch.Tensor) or isinstance(other, float) or isinstance(other, int) or isinstance(other, torch.Tensor):
            return HybridZonotope(self.head / other,
                                  None if self.beta is None else self.beta / abs(other),
                                  None if self.errors is None else self.errors / other,
                                  self.domain)
        else:
            assert False, 'Unknown type of other object'

    def __mul__(self, other: Union[Tensor, int, float]) -> "HybridZonotope":
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, int) or (isinstance(other, torch.Tensor)):
            d = self.head.device
            return HybridZonotope((self.head * other).to(d),
                                    None if self.beta is None else (self.beta * abs(other)).to(d),
                                    None if self.errors is None else (self.errors * other).to(d),
                                    self.domain)
        else:
            assert False, 'Unknown type of other object'

    def __rmul__(self, other):  # Assumes associativity
        return self.__mul__(other)

    def __getitem__(self, indices) -> "HybridZonotope":
        if not isinstance(indices, tuple):
            indices = tuple([indices])
        return HybridZonotope(self.head[indices],
                              None if self.beta is None else self.beta[indices],
                              None if self.errors is None else self.errors[(slice(None), *indices)],
                              self.domain)

    def clone(self) -> "HybridZonotope":
        return HybridZonotope(self.head.clone(),
                              None if self.beta is None else self.beta.clone(),
                              None if self.errors is None else self.errors.clone(),
                              self.domain)

    def detach(self) -> "HybridZonotope":
        return HybridZonotope(self.head.detach(),
                              None if self.beta is None else self.beta.detach(),
                              None if self.errors is None else self.errors.detach(),
                              self.domain)

    def max_center(self) -> Tensor:
        return self.head.max(dim=1)[0].unsqueeze(1)

    def avg_pool2d(self, kernel_size: int, stride:int) -> "HybridZonotope":
        new_head = F.avg_pool2d(self.head, kernel_size, stride)
        new_beta = None if self.beta is None else F.avg_pool2d(self.beta.view(-1, *self.head.shape[1:]), kernel_size, stride)
        new_errors = None if self.errors is None else F.avg_pool2d(self.errors.view(-1, *self.head.shape[1:]), kernel_size, stride).view(-1, *new_head.shape)
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def global_avg_pool2d(self) -> "HybridZonotope":
        new_head = F.adaptive_avg_pool2d(self.head, 1)
        new_beta = None if self.beta is None else F.adaptive_avg_pool2d(self.beta.view(-1, *self.head.shape[1:]), 1)
        new_errors = None if self.errors is None else F.adaptive_avg_pool2d(self.errors.view(-1, *self.head.shape[1:]), 1).view(-1, *new_head.shape)
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def max_pool2d(self, kernel_size, stride):
        if self.errors is not None:
            assert False, "MaxPool for Zono not Implemented"
        lb, ub = self.concretize()
        new_lb = F.max_pool2d(lb, kernel_size, stride)
        new_ub = F.max_pool2d(ub, kernel_size, stride)
        return HybridZonotope.construct_from_bounds(new_lb, new_ub, self.dtype ,self.domain)

    def conv2d(self, weight:Tensor, bias:Tensor, stride:int, padding:int, dilation:int, groups:int) -> "HybridZonotope":
        new_head = F.conv2d(self.head, weight, bias, stride, padding, dilation, groups)
        new_beta = None if self.beta is None else F.conv2d(self.beta, weight.abs(), None, stride, padding, dilation, groups)
        if self.errors is not None:
            errors_resized = self.errors.view(-1, *self.errors.size()[2:])
            new_errors = F.conv2d(errors_resized, weight, None, stride, padding, dilation, groups)
            new_errors = new_errors.view(self.errors.size()[0], self.errors.size()[1], *new_errors.size()[1:])
        else:
            new_errors = None
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def convtranspose2d(self, weight:Tensor, bias:Tensor, stride:int, padding:int,  output_padding:int, dilation:int, groups:int) -> "HybridZonotope":
        new_head = F.conv_transpose2d(self.head, weight, bias, stride, padding, output_padding,  dilation, groups)
        new_beta = None if self.beta is None else F.conv_transpose2d(self.beta, weight.abs(), None, stride, padding, output_padding,  dilation, groups)
        if self.errors is not None:
            errors_resized = self.errors.view(-1, *self.errors.size()[2:])
            new_errors = F.conv_transpose2d(errors_resized, weight, None, stride, padding, output_padding, dilation, groups)
            new_errors = new_errors.view(self.errors.size()[0], self.errors.size()[1], *new_errors.size()[1:])
        else:
            new_errors = None
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def contains(self, other):
        assert(self.head.size(0) == 1)
        if self.errors is None and other.errors is None: #interval
            lb, ub = self.concretize()
            other_lb, other_ub = other.concretize()
            contained = (lb <= other_lb) & (other_ub <= ub)
            cont_factor = 2*torch.max(((other_ub-self.head)/(ub-lb+1e-16)).abs().max(), ((other_lb-self.head)/(ub-lb+1e-16)).abs().max())
            return contained.all(), cont_factor
        else:
            raise NotImplementedError()
            # lb, ub = self.concretize()
            # other_lb, other_ub = other.concretize()
            # contained = (lb <= other_lb) & (other_ub <= ub)
            # return contained.all()

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
                              None if self.errors is None else self.errors.matmul(other),
                              self.domain)

    def rev_matmul(self, other: Tensor) -> "HybridZonotope":
        return HybridZonotope(other.matmul(self.head),
                              None if self.beta is None else other.abs().matmul(self.beta),
                              None if self.errors is None else other.matmul(self.errors),
                              self.domain)

    def fft(self):
        assert self.beta is None
        return HybridZonotope(torch.fft.fft2(self.head).real,
                         None,
                         None if self.errors is None else torch.fft.fft2(self.errors).real,
                         self.domain)

    @staticmethod
    def cat(zonos, dim=0):
        new_head = torch.cat([x.head for x in zonos], dim)
        new_beta = torch.cat([x.beta if x.beta is not None else torch.zeros_like(x.head) for x in zonos], dim)
        dtype = zonos[0].head.dtype
        device = zonos[0].head.device

        errors = [zono.errors for zono in zonos if zono.errors is not None]
        if len(errors)>0:
            n_err = [x.shape[0] for x in errors]

            new_errors = torch.zeros((n_err, *new_head.shape), dtype=dtype, device=device).transpose(1, dim+1)

            i=0
            j=0
            for error in errors:
                error = error.transpose(1, dim+1)
                new_errors[i:i+error.shape[0], j:j+error.shape[1+dim]] = error
                i += error.shape[0]
                j += error.shape[1+dim]
            new_errors = new_errors.transpose(1, dim + 1)

        else:
            new_errors = None

        return HybridZonotope(new_head, new_beta, new_errors, zonos[0].domain)


    def relu(self, deepz_lambda, bounds, init_lambda):
        if self.errors is None:
            min_relu, max_relu = F.relu(self.head - self.beta), F.relu(self.head + self.beta)
            return HybridZonotope(0.5 * (max_relu + min_relu), 0.5 * (max_relu - min_relu), None, self.domain), None
        assert self.beta is None
        delta = torch.sum(torch.abs(self.errors), 0)
        lb, ub = self.head - delta, self.head + delta

        if bounds is not None:
            lb_refined, ub_refined = bounds
            lb = torch.max(lb_refined, lb)
            ub = torch.min(ub_refined, ub)
        is_cross = (lb < 0) & (ub > 0)

        D = 1e-6
        relu_lambda = torch.where(is_cross, ub/(ub-lb+D), (lb >= 0).float())
        if self.domain == 'zono_iter':
            if init_lambda:
                # print(relu_lambda.size())
                # print(deepz_lambda.size())
                deepz_lambda.data = relu_lambda.data.squeeze(0)

            assert (deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()

            relu_lambda_cross = deepz_lambda.unsqueeze(0)
            relu_mu_cross = torch.where(relu_lambda_cross < relu_lambda, 0.5*ub*(1-relu_lambda_cross), -0.5*relu_lambda_cross*lb)

            # relu_lambda_cross = deepz_lambda * relu_lambda
            # relu_mu_cross = 0.5*ub*(1-relu_lambda_cross)

            # relu_lambda_cross = relu_lambda + (1 - deepz_lambda) * (1 - relu_lambda)
            # relu_mu_cross = -0.5*relu_lambda_cross*lb

            relu_lambda = torch.where(is_cross, relu_lambda_cross, (lb >= 0).float())
            relu_mu = torch.where(is_cross, relu_mu_cross, torch.zeros(lb.size()).to(self.device))
        else:
            relu_mu = torch.where(is_cross, -0.5*ub*lb/(ub-lb+D), torch.zeros(lb.size()).to(self.device))

        assert (not torch.isnan(relu_mu).any()) and (not torch.isnan(relu_lambda).any())

        new_head = self.head * relu_lambda + relu_mu
        old_errs = self.errors * relu_lambda
        new_errs = get_new_errs(is_cross, new_head, relu_mu)
        new_errors = torch.cat([old_errs, new_errs], dim=0)
        assert (not torch.isnan(new_head).any()) and (not torch.isnan(new_errors).any())
        return HybridZonotope(new_head, None, new_errors, self.domain), None

    def concretize(self):
        delta = 0
        if self.beta is not None:
            delta = delta + self.beta
        if self.errors is not None:
            delta = delta + self.errors.abs().sum(0)
        return self.head - delta, self.head + delta

    def avg_width(self):
        lb, ub = self.concretize()
        return (ub - lb).mean()

    def is_greater(self, i, j):
        if self.errors is not None:
            diff_errors = (self.errors[:, :, i] - self.errors[:, :, j]).abs().sum(dim=0)
            diff_head = self.head[:, i] - self.head[:, j]
            delta = diff_head - diff_errors
            if self.beta is not None:
                delta -= self.beta[:, i].abs() + self.beta[:, j].abs()
            return delta, delta > 0
        else:
            diff_head = (self.head[:, i] - self.head[:, j])
            diff_beta = (self.beta[:, i] + self.beta[:, j]).abs()
            delta = (diff_head - diff_beta)
            return delta, delta > 0

    def verify(self, targets):
        n_class = self.head.size()[1]
        verified = torch.zeros(targets.size(), dtype=torch.uint8).to(self.head.device)
        verified_corr = torch.zeros(targets.size(), dtype=torch.uint8).to(self.head.device)
        for i in range(n_class):
            isg = torch.ones(targets.size(), dtype=torch.uint8).to(self.head.device)
            for j in range(n_class):
                if i != j:
                    _, ok = self.is_greater(i, j)
                    isg = isg & ok.byte()
            verified = verified | isg
            verified_corr = verified_corr | (targets.eq(i).byte() & isg)
        return verified, verified_corr

    def get_min_diff(self, i, j):
        """ returns minimum of logit[i] - logit[j] """
        return self.is_greater(i, j)[0]

    def get_wc_logits(self, targets):
        batch_size = targets.size()[0]
        lb, ub = self.concretize()
        wc_logits = ub
        wc_logits[np.arange(batch_size), targets] = lb[np.arange(batch_size), targets]
        return wc_logits

    def ce_loss(self, targets):
        wc_logits = self.get_wc_logits(targets)
        return F.cross_entropy(wc_logits, targets)

    def subpatch(self, start, size):
        s = min(start+size, self.head.size(0))
        return HybridZonotope(self.head[start:s, ...],
                              None if self.beta is None else self.beta[start:s, ...],
                              None if self.errors is None else self.errors[:, start:s, ...],
                              self.domain)

    def to(self, device):
        return HybridZonotope(self.head.to(device),
                              None if self.beta is None else self.beta.to(device),
                              None if self.errors is None else self.errors.to(device),
                              self.domain)

    def apply_block(self, block):
        return apply_block(self, block)

if __name__ == '__main__':
    img = 0.5 * torch.ones((1, 2, 3, 3))
    img[0, 0, 0, 1] = 0.9
    h = HybridZonotope.zonotope_patch_from_noise(img, 0.3, 'zono', pos='all')
