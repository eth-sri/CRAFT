"""
Based on HybridZonotope from DiffAI (https://github.com/eth-sri/diffai/blob/master/ai.py)
"""
import uuid
import numpy as np
import torch
import torch.nn.functional as F
import math
from utils import basis_transform, get_new_basis
from monDEQ.mon import fft_conv
from ai.ai_util import AbstractElement


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
            curr, _ = curr.relu(None, None, None)
        else:
            print(type(l))
            assert(False)
    return curr


def clamp_image(x, eps, data_min=0, data_max=1):
    if isinstance(eps, torch.Tensor):
        assert(len(eps.size()) == 1 and eps.size(0) == x.size(0))
        s = [1] * len(x.size())
        s[0] = eps.size(0)
        eps = eps.view(s)
    min_x = x-eps if data_min is None else torch.clamp(x-eps, min=data_min)
    max_x = x + eps if data_max is None else torch.clamp(x+eps, max=1)
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


def get_eye_casted(shape):
    """ Generates the 'identity' for this matrix i.e. it is written that for every eps value in the first dimension we select exactly one entry from the rest of the matrix

    Args:
        shape ([type]): The shape we want to generate over

    Returns:
        [type]: The identity
    """
    # Setup variables
    m = shape[0]
    n = math.prod(shape[1:])
    
    # Switch in case m < n
    transpose = False
    if m < n: 
        transpose = True
        m, n = n, m
    
    # Can assume m >= n
    e = torch.eye(n)
    t = torch.tile(e, (math.ceil(m/n), 1))
    t = t[:m, :]

    if transpose:
        t = t.t()
    
    res = t.reshape(shape)
    return res


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


def expand_to(vec1, to_vec):
    if vec1.shape == to_vec.shape:
        return vec1
    else:
        expand_len = torch.ones(len(to_vec.shape))
        expand_len[0] = to_vec.shape[0]
        expand_len = expand_len.int()

        return vec1.view(expand_len.tolist()).expand_as(to_vec)


def add_zono_errors(zono_errors_1, zono_errors_2):
    if zono_errors_1 == None:
        return zono_errors_2
    elif zono_errors_2 == None:
        return zono_errors_1
    else:
        new_dict = {}
        for key1, val1 in zono_errors_1.items():
            val2 = zono_errors_2.get(key1)
            if val2 is not None:
                new_dict[key1] = val1 + val2
            else:
                new_dict[key1] = val1

        for key2, val2 in zono_errors_2.items():
            contained = new_dict.get(key2)
            if contained is None:
                new_dict[key2] = val2

    return new_dict


class LBZonotope(AbstractElement):
    def __init__(self, head, errors, box_errors, domain="LBZonotope", error_id=None):
        self.head = head

        # Handle zono_errors
        if isinstance(errors, torch.Tensor):
            # New zonotope
            self.zono_errors = {}
            self.add_zono_error(errors, error_id)
        elif isinstance(errors, dict):
            # Already in correct form
            self.zono_errors = errors
        elif errors is None:
            self.zono_errors = {}
        else:
            assert False, f"Errors with type {type(errors)} are not supported."
        self.zono_errors_inv = None

        if isinstance(box_errors, torch.Tensor):
            # New zonotope
            self.add_box_error(box_errors)
        elif isinstance(box_errors, dict):
            # Already in correct form
            self.box_errors = box_errors
        elif box_errors is None:
            self.box_errors = None
        else:
            assert False, f"Errors with type {type(errors)} are not supported."

        self.domain = domain
        self.dimension = self.head.size()[-1]
        assert not torch.isnan(self.head).any()
        assert self.zono_errors is None or (not any([torch.isnan(x).any() for x in self.zono_errors.values()]))
        # assert self.zono_errors is not None

    @staticmethod
    def lb_zonotope_from_noise(x, eps, error_id=None, domain='zono', dtype=None, data_min=0, data_max=1):
        if dtype is None:
            dtype=x.dtype
        ei = LBZonotope.get_error_matrix(x, dtype)
        x_center, x_beta = clamp_image(x, eps, data_min, data_max)
        x_center, x_beta = x_center.to(dtype=dtype), x_beta.to(dtype=dtype)
        return LBZonotope(x_center, ei * x_beta.unsqueeze(0), None, error_id=error_id)

    @staticmethod
    def lb_zonotope_from_bounds(specLB, specUB, dtype=None, data_min=0, data_max=1):
        ei = LBZonotope.get_error_matrix(specLB, dtype)
        specLB, specUB = torch.clamp(specLB, min=data_min, max=data_max), torch.clamp(specUB, min=data_min, max=data_max)
        x_center = (specLB + specUB) / 2.
        x_beta = (specUB - specLB) / 2.
        x_center, x_beta = x_center.to(dtype=dtype), x_beta.to(dtype=dtype)
        return LBZonotope(x_center, ei * x_beta.unsqueeze(0), None)

    @staticmethod
    def get_error_matrix(x, dtype=None, error_idx=None):
        if dtype is None:
            dtype = x.dtype
        batch_size = x.size()[0]
        if error_idx is None:
            n_elements_e = x[0].numel()
            ei = torch.eye(n_elements_e, dtype=dtype, device=x.device).expand(batch_size, n_elements_e, n_elements_e).permute(1, 0, 2)
        else:
            assert batch_size == 1
            n_elements_e = int(error_idx.sum())
            n_elements_x = x[0].numel()
            ei = torch.zeros((n_elements_e, n_elements_x), dtype=dtype, device=x.device)
            ei[torch.arange(n_elements_e).view(-1,1), error_idx.flatten().nonzero()] = 1
            ei = ei.expand(batch_size, n_elements_e, n_elements_x).permute(1, 0, 2)
        if len(x.size()) > 2:
            ei = ei.contiguous().view(n_elements_e, *x.size())
        return ei

    def add_zono_error(self, errors, error_id=None):
        error_id = uuid.uuid4() if error_id is None else error_id
        if not hasattr(self, "zono_errors") or self.zono_errors is None:
            self.zono_errors = {}
        new_errors = errors #if isinstance(errors,tuple) else (errors,  torch.ones(errors.shape[0], dtype=self.head.dtype, device=self.head.device))
        self.zono_errors[error_id] = new_errors

    def add_box_error(self, errors, error_id=None, consolidate=True):
        error_id = uuid.uuid4() if error_id is None else error_id
        if errors is None:
            return
        elif errors.shape != self.head.shape:
            errors = errors.abs().sum(0)

        if consolidate:
            errors += self.concretize("box")[1]

        errors = self.get_error_matrix(self.head, error_idx=errors != 0) * errors.unsqueeze(0)

        if consolidate:
            # self.box_errors = {error_id: (errors,  torch.ones(errors.shape[0], dtype=self.head.dtype, device=self.head.device))}
            self.box_errors = {error_id: errors}
        else:
            if self.box_errors is None:
                self.box_errors = {}
            # self.box_errors[error_id] = (errors,  torch.ones(errors.shape[0], dtype=self.head.dtype, device=self.head.device))
            self.box_errors[error_id] = errors

    def box_errors_to_zono(self):
        if self.box_errors is None:
            return
        for key, value in self.box_errors.items():
            self.add_zono_error(value, key)
        self.box_errors = None

    def consolidate_box_errors(self, shift_to_zono=True):
        '''
        Consolidates box terms to obtain at most one box term.
        If shift_to_zono is enabled, box_errors that are non_diagonal are moved to zono_errors, and this does not cause
        a loss of precision, else a box approximation of all box_errors is used
        :param shift_to_zono: Allow the generation of zono_errors
        :return: LBZonotope with at most one box_errors entry
        '''
        if self.box_errors is None:
            return

        if shift_to_zono:
            ei = self.get_error_matrix(self.head)
            diag_terms = None
            for key, val in self.box_errors.items():
                concretized_box_terms = val.abs().sum(0)
                if torch.isclose(ei * concretized_box_terms.unsqueeze(0), val.abs()).all():
                    # error matrix is diagonal
                    if diag_terms is None:
                        diag_terms = concretized_box_terms
                    else:
                        diag_terms += concretized_box_terms
                else:
                    self.add_zono_error(val, key)
        else:
            diag_terms = self.concretize("box")

        self.box_errors = None          # Set current box errors to 0
        self.add_box_error(diag_terms)  # Add consolidated box errors

    def consolidate_errors(self, new_basis_vectors=None, basis_transform_method="pca", verbose=False, error_eps_add=1e-8, error_eps_mul=1e-8):
        if new_basis_vectors is None:
            errors_to_get_basis_from = self.get_zono_matrix()
            shp = errors_to_get_basis_from.shape
            errors_to_get_basis_from = errors_to_get_basis_from.view(shp[0], -1)

            new_basis_vectors = get_new_basis(errors_to_get_basis_from, basis_transform_method)
            if verbose:
                old_rank_of_errors = np.linalg.matrix_rank(new_basis_vectors)
                new_rank_of_errors = np.linalg.matrix_rank(new_basis_vectors)
                print(f"New error basis computed - Rank of Errors {old_rank_of_errors}/{new_rank_of_errors}")

        return self.move_to_basis(new_basis_vectors, error_eps_add, error_eps_mul), new_basis_vectors

    def size(self):
        return self.head.size()

    @staticmethod
    def zeros(*args, **kwargs):
        zero = torch.zeros(*args, **kwargs)
        return LBZonotope(zero, None, None, "LBZonotope")

    def view(self, size):
        if self.zono_errors is not None:
            new_zono_errors = {}
            for key, val in self.zono_errors.items():
                if val.size()[0] == 0: continue
                new_zono_errors[key] = val.view(val.size()[0], *size)
        else:
            new_zono_errors = None

        if self.box_errors is not None:
            new_box_errors = {}
            for key, val in self.box_errors.items():
                if val.size()[0] == 0: continue
                new_box_errors[key] = val.view(val.size()[0], *size)
        else:
            new_box_errors = None

        return LBZonotope(self.head.view(*size), new_zono_errors, new_box_errors, self.domain)

    def flatten(self):
        bsize = self.head.size(0)
        return self.view((bsize, -1))

    def normalize(self, mean, sigma):
        return (self - mean) / sigma

    def __sub__(self, other):
        if isinstance(other, torch.Tensor) or isinstance(other, float) or isinstance(other, int) :
            return LBZonotope(self.head - other, self.zono_errors, self.box_errors, self.domain)
        elif isinstance(other, LBZonotope):
            return self + (-1 * other)
        else:
            assert False, 'Unknown type of other object'

    def __add__(self, other): # NOTE: Only use if you know what to expect 
        if isinstance(other, torch.Tensor):
            return LBZonotope(self.head + other, self.zono_errors, self.box_errors, self.domain)
        elif isinstance(other, LBZonotope):
            res_zono_errors = add_zono_errors(self.zono_errors, other.zono_errors)
            res_box_errors = add_zono_errors(self.box_errors, other.box_errors)
            return LBZonotope(self.head + other.head, res_zono_errors, res_box_errors)
        else:
            assert False, 'Unknown type of other object'

    def __truediv__(self, other):
        if isinstance(other, torch.Tensor):
            return LBZonotope(self.head / other,
                              None if self.zono_errors is None else map_dict(self.zono_errors, lambda x: x / other),
                              None if self.box_errors is None else map_dict(self.box_errors, lambda x: x / other), self.domain)
        else:
            assert False, 'Unknown type of other object'

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float) or (isinstance(other, torch.Tensor) and (other.shape == self.shape or other.shape == torch.Size([1]) or other.shape == torch.Size([]))):
            d = self.head.device
            return LBZonotope(self.head * other,
                              None if self.zono_errors is None else map_dict(self.zono_errors, lambda x: x * other),
                              None if self.box_errors is None else map_dict(self.box_errors, lambda x: x * other),
                              self.domain)
        else:
            assert False, 'Unknown type of other object'

    def __rmul__(self, other):  # Assumes associativity of multiplication
        return self.__mul__(other)

    def clone(self):
        return LBZonotope(self.head.clone(),
                          None if self.zono_errors is None else map_dict(self.zono_errors, lambda x: x.clone()),
                          None if self.box_errors is None else map_dict(self.box_errors, lambda x: x.clone()),
                          self.domain)

    def detach(self):
        return LBZonotope(self.head.detach(),
                          None if self.zono_errors is None else map_dict(self.zono_errors, lambda x: x.detach()),
                          None if self.box_errors is None else map_dict(self.box_errors, lambda x: x.detach()),
                          self.domain)

    def __getitem__(self, item):
        return LBZonotope(self.head[item],
                          None if self.zono_errors is None else map_dict(self.zono_errors, lambda x: x[(slice(None),)+tuple(item)]),
                          None if self.box_errors is None else map_dict(self.box_errors, lambda x: x[(slice(None),)+tuple(item)]),
                          self.domain)

    @property
    def shape(self):
        return self.head.shape

    @property
    def device(self):
        return self.head.device

    @property
    def dtype(self):
        return self.head.dtype

    @staticmethod
    def cat(zonos, dim=0):
        new_head = torch.cat([x.head for x in zonos], dim)
        zono_error_keys = []
        box_error_keys = []
        dtype = zonos[0].head.dtype
        device = zonos[0].head.device

        for zono in zonos:
            zono_error_keys += list(zono.zono_errors.keys()) if zono.zono_errors is not None else []
            box_error_keys += list(zono.box_errors.keys()) if zono.box_errors is not None else []
        zono_error_keys = list(set(zono_error_keys))
        box_error_keys = list(set(box_error_keys))

        new_zono = LBZonotope(new_head, None, None, zonos[0].domain)

        for key in zono_error_keys:
            n_err = [x.zono_errors[key].shape[0] for x in zonos if key in x.zono_errors.keys()][0]
            new_error = torch.cat([x.zono_errors[key] if key in x.zono_errors.keys() else
                                   torch.zeros((n_err, *x.shape), dtype=dtype, device=device) for x in zonos], dim+1)
            new_zono.add_zono_error(new_error, error_id=key)

        for key in box_error_keys:
            n_err = [x.box_errors[key].shape[0] for x in zonos if (x.box_errors is not None and key in x.box_errors.keys())][0]
            new_error = torch.cat([x.box_errors[key] if (x.box_errors is not None and key in x.box_errors.keys())
                                   else torch.zeros((n_err, *x.shape), dtype=dtype, device=device) for x in zonos], dim+1)
            new_zono.add_box_error(new_error, error_id=key, consolidate=False)

        return new_zono

    def avg_pool2d(self, kernel_size, stride):
        new_head = F.avg_pool2d(self.head, kernel_size, stride)
        new_zono_errors = None if self.zono_errors is None else map_dict(self.zono_errors, lambda x: F.avg_pool2d(x.view(-1, *self.head.shape[1:]), kernel_size, stride))
        new_box_errors = None if self.box_errors is None else map_dict(self.box_errors, lambda x: F.avg_pool2d(x.view(-1, *self.head.shape[1:]), kernel_size, stride))
        return LBZonotope(new_head, new_zono_errors, new_box_errors, self.domain)

    def pad(self, padding, mode):
        new_head = F.pad(self.head, padding, mode=mode)

        zono_errors_resized = map_dict(self.zono_errors, lambda x: x.view(-1, *x.size()[2:]))
        new_zono_errors     = map_dict(zono_errors_resized, lambda x: F.pad(x, padding, mode=mode))
        res = {}
        for key, val in new_zono_errors.items():
            res[key] = val.view(self.zono_errors[key].size()[0], self.zono_errors[key].size()[1], *new_head.shape[1:])
        new_zono_errors = res

        if self.box_errors is None:
            new_box_errors = None
        else:
            box_errors_resized = map_dict(self.box_errors, lambda x: x.view(-1, *x.size()[2:]))
            new_box_errors = map_dict(box_errors_resized, lambda x: F.pad(x, padding, mode=mode))
            res = {}
            for key, val in new_box_errors.items():
                res[key] = val.view(self.box_errors[key].size()[0], self.box_errors[key].size()[1], *new_head.shape[1:])
            new_box_errors = res

        return LBZonotope(new_head, new_zono_errors, new_box_errors, self.domain)

    def conv2d(self, weight, bias, stride, padding, dilation, groups):
        new_head = F.conv2d(self.head, weight, bias, stride, padding, dilation, groups)

        zono_errors_resized = map_dict(self.zono_errors, lambda x: x.view(-1, *x.size()[2:]))
        new_zono_errors     = map_dict(zono_errors_resized, lambda x: F.conv2d(x, weight, None, stride, padding, dilation, groups))
        res = {}
        for key, val in new_zono_errors.items():
            res[key] = val.view(self.zono_errors[key].size()[0], self.zono_errors[key].size()[1], *new_head.shape[1:])
        new_zono_errors = res

        if self.box_errors is None:
            new_box_errors = None
        else:
            box_errors_resized = map_dict(self.box_errors, lambda x: x.view(-1, *x.size()[2:]))
            new_box_errors = map_dict(box_errors_resized, lambda x: F.conv2d(x, weight, None, stride, padding, dilation, groups))
            res = {}
            for key, val in new_box_errors.items():
                res[key] = val.view(self.box_errors[key].size()[0], self.box_errors[key].size()[1], *new_head.shape[1:])
            new_box_errors = res

        return LBZonotope(new_head, new_zono_errors, new_box_errors, self.domain)
    
    def conv_transpose2d(self, weight, bias, stride, padding, output_padding, groups, dilation):
        new_head = F.conv_transpose2d(self.head, weight, bias, stride, padding, output_padding, groups, dilation)

        zono_errors_resized = map_dict(self.zono_errors, lambda x: x.view(-1, *x.size()[2:]))
        new_zono_errors     = map_dict(zono_errors_resized, lambda x: F.conv_transpose2d(x, weight, None, stride, padding, output_padding,  groups, dilation))
        res = {}
        for key, val in new_zono_errors.items():
            res[key] = val.view(self.zono_errors[key].size()[0], self.zono_errors[key].size()[1], *new_head.shape[1:])
        new_zono_errors = res

        if self.box_errors is None:
            new_box_errors = None
        else:
            box_errors_resized = map_dict(self.box_errors, lambda x: x.view(-1, *x.size()[2:]))
            new_box_errors = map_dict(box_errors_resized, lambda x: F.conv_transpose2d(x, weight, None, stride, padding, output_padding,  groups, dilation))
            res = {}
            for key, val in new_box_errors.items():
                res[key] = val.view(self.box_errors[key].size()[0], self.box_errors[key].size()[1], *new_head.shape[1:])
            new_box_errors = res
        return LBZonotope(new_head, new_zono_errors, new_box_errors, self.domain)

    def fft_conv(self, w_fft, transpose=False):
        new_head = fft_conv(self.head, w_fft, transpose)

        zono_errors_resized = map_dict(self.zono_errors, lambda x: x.view(-1, *x.size()[2:]))
        new_zono_errors = map_dict(zono_errors_resized, lambda x: fft_conv(x, w_fft, transpose))
        res = {}
        for key, val in new_zono_errors.items():
            res[key] = val.view(self.zono_errors[key].size()[0], self.zono_errors[key].size()[1], *new_head.shape[1:])
        new_zono_errors = res

        if self.box_errors is None:
            new_box_errors = None
        else:
            box_errors_resized = map_dict(self.box_errors, lambda x: x.view(-1, *x.size()[2:]))
            new_box_errors = map_dict(box_errors_resized, lambda x:  fft_conv(x, w_fft, transpose))
            res = {}
            for key, val in new_box_errors.items():
                res[key] = val.view(self.box_errors[key].size()[0], self.box_errors[key].size()[1], *new_head.shape[1:])
            new_box_errors = res
        return LBZonotope(new_head, new_zono_errors, new_box_errors, self.domain)

    def contains(self, other, centered=False, verbose=False):
        dtype = self.head.dtype
        device = self.head.device

        # Solve the LGS that we get when representing the "other" zonotope in the "self" basis

        # System Ax = B
        # NOTE: This gives us eps parameterized vectors in x space (i.e. shape 40x824)
        A = self.get_zono_matrix().flatten(start_dim=1).T # containing init errors
        B = other.get_zono_matrix().flatten(start_dim=1).T # contained init errors

        if A.shape[-1] == A.shape[-2] and self.zono_errors_inv is None:
            try:
                self.zono_errors_inv = torch.inverse(A)
            except:
                print("Failed to invert error matrix")

        if self.zono_errors_inv is None:
            if A.shape[0] != A.shape[1]:
                sol = np.linalg.lstsq(A.cpu().numpy(), B.cpu().numpy(), rcond=None)
                x = torch.tensor(sol[0], dtype=dtype, device=device)
            elif float(torch.__version__[:-2]) < 1.9:
                x = torch.solve(B, A).solution
            else:
                x = torch.linalg.solve(A, B)
        else:
            x = torch.matmul(self.zono_errors_inv, B)

        # Note sometimes we dont have full rank for A (check sol[1]) - the solution however has no residuals
        # Here x contains the coordinates of the inner zonotope in the outer basis -> 824 x 824

        if not torch.isclose(torch.matmul(A, x), B, atol=1e-7, rtol=1e-6).all(): #, f"Projection of contained errors into base of containing errors failed"
            uncaptured_errors = torch.abs(B-torch.matmul(A, x)).sum(axis=1)
            # assert False
        else:
            uncaptured_errors = torch.zeros_like(self.head)

        # Sum the absolutes row-wise to get the scaling factor for the containing error coefficients to overapproximated the contained ones
        abs_per_orig_vector = torch.sum(torch.abs(x), axis=1)
        max_sp = torch.max(abs_per_orig_vector).cpu().item()

        if max_sp > 1 or str(max_sp) == "nan":
            if verbose:
                print(f"Containment of init errors failed with {max_sp}")
            return False, max_sp

        # Here would could probably do some smarter combination i.e. we could compensate the worst errors of the init errors in the differences in the merge errors
        # However this is generally hard (I believe) - a greedy solution should work

        # Here we adjust for the head displacement
        diff = torch.abs(self.head-other.head).detach().view(-1)

        # Here we adjust for errors not captured by the intial matching due to differences in spanned space:
        diff += uncaptured_errors.view(-1)

        # Here we just do a basic check on the "independent" merge errors
        if other.box_errors is None:
            max_sp_merge = 0
            merge_cont = True
        elif self.box_errors is None:
            max_sp_merge = "nan"
            merge_cont = False
        else:
            merge_cont = True
            # ei = self.get_error_matrix(self.head)
            for key, val in self.box_errors.items():
                # Ensure that all box_errors of the outer zono actually induce a box
                if not ((val != 0).sum(0) <= 1).all(): #
                    merge_cont = False
                    max_sp_merge = "nan"

            # Check that merge errors (or rather their difference) is diagonal
            if merge_cont:
                self_box_errors = self.concretize("box")[1].detach()
                other_box_errors = other.concretize("box")[1].detach()

                merge_diff = (self_box_errors - other_box_errors).detach().view(-1)
                merge_cont = (merge_diff >= 0).all()    # Ensure that the box errors of other can be contained with the box errors of selfe
                max_sp_merge = torch.max(other_box_errors / (self_box_errors + 1e-8)).cpu().item()

                # When the merge errors of the containing zono are larger than that of the contained one, we can use this extra to compensate for some of the difference in the heads
                # diff = torch.maximum(diff - torch.diagonal(merge_diff), torch.tensor(0)).detach()
                diff = torch.maximum(diff - merge_diff, torch.zeros_like(diff)).detach()

        if not merge_cont:
            if verbose:
                print(f"Containment of merge errors failed")
            return False, max_sp_merge

        # This projects the remaining difference between the heads into the error coefficient matrix
        diff = torch.diag(diff.view(-1))
        if self.zono_errors_inv is None:
            if A.shape[0] != A.shape[1]:
                sol_diff = np.linalg.lstsq(A.cpu().numpy(), diff.cpu().numpy(), rcond=None)
                x_diff = torch.tensor(sol_diff[0], dtype=dtype, device=device)
            elif float(torch.__version__[:-2]) < 1.9:
                x_diff = torch.solve(diff, A).solution
            else:
                x_diff = torch.linalg.solve(A, diff)
        else:
            x_diff = torch.matmul(self.zono_errors_inv, diff)


        if not torch.isclose(torch.matmul(A, x_diff), diff, atol=1e-7, rtol=1e-6).all():
            #f"Projection of head difference into base of containing errors failed"
            return False, np.inf

        abs_per_orig_vector_diff = abs_per_orig_vector + torch.abs(x_diff).sum(axis=1)
        max_sp_diff = torch.max(abs_per_orig_vector_diff).cpu().item()

        # Check if with this additional component, we are still contained
        if max_sp_diff > 1 or str(max_sp_diff) == "nan":
            if verbose:
                print(f"Containment of head differences failed with {max_sp_diff}")
            return False, max_sp_diff

        if verbose:
            print(f"Containment with {max_sp_diff}")

        return True, max(max_sp_merge, max_sp_diff, max_sp)


    def move_to_basis(self, new_basis, ERROR_EPS_ADD=0, ERROR_EPS_MUL=0):
        # This moves the init errors of a zonotope into a new basis (and essentially reduces the number of error terms via over-approximation)
        dtype = self.head.dtype
        device = self.head.device

        if self.zono_errors is None:
            return LBZonotope(self.head, None, self.box_errors, self.domain, -1)
        else:
            # Old basis is the concatenation of all zono_errors
            old_error_terms = self.get_zono_matrix()
            shp = old_error_terms.shape

            old_error_terms = old_error_terms.view(shp[0], -1).cpu().numpy().T
            new_zono_error_terms = basis_transform(new_basis, old_error_terms, ERROR_EPS_ADD, ERROR_EPS_MUL, return_full_res=False).to(device)

            return LBZonotope(self.head, new_zono_error_terms.view(-1,1,*shp[1:]), self.box_errors, self.domain)
        

    def get_zono_matrix(self):
        if self.zono_errors is None:
            return None
        return torch.cat(list(self.zono_errors.values())).detach().squeeze()

    def get_box_matrix(self):
        if self.box_errors is None or list(self.box_errors.values())==[]:
            return None
        return torch.cat(list(self.box_errors.values())).detach().squeeze()

    def linear(self, weight, bias=None):
        ret = self.matmul(weight.t())
        if bias is not None:
            ret = ret + bias.unsqueeze(0)
        return ret

    def matmul(self, other):
        return LBZonotope(self.head.matmul(other),
                          None if self.zono_errors is None else map_dict(self.zono_errors, lambda x: x.matmul(other)),
                          None if self.box_errors is None else map_dict(self.box_errors, lambda x: x.matmul(other)),
                          self.domain)

    def batch_norm(self, bn) -> "LBZonotope":
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
        new_zono_errors = None if self.zono_errors is None else map_dict(self.zono_errors, lambda x: x * c.view(*([1]+view_dim_list)))
        new_box_errors = None if self.box_errors is None else map_dict(self.box_errors, lambda x: x * c.view(*([1]+view_dim_list)))
        return LBZonotope(new_head, new_zono_errors, new_box_errors, self.domain)

    def relu(self, deepz_lambda, bounds, init_lambda=False):
        dtype = self.head.dtype
        device = self.head.device
        D = 1e-8

        lb, ub = self.concretize()
        if bounds is not None:
            lb_refined, ub_refined = bounds
            lb = torch.max(lb_refined, lb)
            ub = torch.min(ub_refined, ub)
        is_cross = (lb < 0) & (ub > 0)

        relu_lambda = ub/(ub-lb+D)
        relu_mu = -0.5 * lb * relu_lambda

        if deepz_lambda is not None:
            if (deepz_lambda < 0).all():
                deepz_lambda.data = relu_lambda.detach()
            relu_mu = torch.where(deepz_lambda < relu_lambda, 0.5 * ub * (1 - deepz_lambda), -0.5 * lb * deepz_lambda)
            relu_lambda = torch.clamp(deepz_lambda, 0, 1)

        relu_mu_cross = torch.where(is_cross.detach(), relu_mu, torch.zeros(lb.size(), dtype=dtype, device=device))
        relu_lambda_cross = torch.where(is_cross.detach(), relu_lambda, (lb >= 0).to(dtype))

        assert (not torch.isnan(relu_mu_cross).any()) and (not torch.isnan(relu_lambda_cross).any())

        # Update the head
        new_head = self.head * relu_lambda_cross + relu_mu_cross

        # Update the zono_errors
        new_zono_errors = map_dict(self.zono_errors, lambda x: x * relu_lambda_cross)

        new_zono = LBZonotope(new_head, new_zono_errors, None, self.domain)
        new_zono.add_box_error(relu_mu_cross)  # Add the new errors

        # Move the old merge errors into the init errors
        if self.box_errors is not None:
            new_zono.zono_errors.update(map_dict(self.box_errors, lambda x: x * relu_lambda_cross))

        return new_zono, deepz_lambda

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

        zono_errors_resized = map_dict(new_zono.zono_errors, lambda x: x.view(-1, *x.size()[2:]))
        new_zono_errors     = map_dict(zono_errors_resized, lambda x: set_boarder_zero(x, border))
        res = {}
        for key, val in new_zono_errors.items():
            res[key] = val.view(new_zono.zono_errors[key].size()[0], new_zono.zono_errors[key].size()[1], *new_head.shape[1:])
        new_zono_errors = res

        if new_zono.box_errors is None:
            new_box_errors = None
        else:
            box_errors_resized = map_dict(new_zono.box_errors, lambda x: x.view(-1, *x.size()[2:]))
            new_box_errors = map_dict(box_errors_resized, lambda x: set_boarder_zero(x, border))
            res = {}
            for key, val in new_box_errors.items():
                res[key] = val.view(new_zono.box_errors[key].size()[0], new_zono.box_errors[key].size()[1], *new_head.shape[1:])
            new_box_errors = res
        return LBZonotope(new_head, new_zono_errors, new_box_errors, new_zono.domain)

    def concretize(self, mode="all"):
        delta = 0
        if self.zono_errors is not None and mode in ["all", "zono"]:
            for val in self.zono_errors.values():
                delta = delta + val.abs().sum(0)
        if self.box_errors is not None and mode in ["all", "box"]:
            for val in self.box_errors.values():
                delta = delta + val.abs().sum(0)
        return self.head * (mode == "all") - delta, self.head * (mode == "all") + delta

    def avg_width(self):
        lb, ub = self.concretize()
        return (ub - lb).mean()

    def to(self, device):
        return LBZonotope(self.head.to(device),
                          None if self.zono_errors is None else map_dict(self.zono_errors, lambda x: x.to(device)),
                          None if self.box_errors is None else map_dict(self.box_errors, lambda x: x.to(device)),
                          self.domain)

    def apply_block(self, block):
        return apply_block(self, block)

if __name__ == '__main__':
    img = 0.5 * torch.ones((1, 2, 3, 3))
    img[0, 0, 0, 1] = 0.9
    h = LBZonotope.zonotope_patch_from_noise(img, 0.3, 'zono', pos='all')
