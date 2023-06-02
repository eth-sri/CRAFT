import os
import re

from typing import Tuple

import torch
import numpy as np
import random
import uuid
import matplotlib.pyplot as plt
from ai.new_zonotope import HybridZonotope


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def cuda(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cuda()
    else:
        return torch.tensor(tensor).cuda()


def expand_args(defaults=None, kwargs=None):
    if defaults is None:
        d = {}
    else:
        d = defaults.copy()
    if kwargs is not None:
        for k, v in kwargs.items():
            d[k] = v
    return d


def get_unique_id():
    return uuid.uuid4()


def check_soundness_by_sampling(z_inner, z_outer, n=20, verbose=False, DTYPE_TORCH=torch.float64):
    """
    Sample random corners of the old zonotope and ensure that error-terms in the new zonotope can be found to describe the same point
    :param z_inner: Inner (M-)Zonotope
    :param z_outer: Outer (M-)Zonotope
    :param n: Number of points to sample
    :param verbose: Whether to print info even when no violation is found
    :param DTYPE_TORCH:
    :return: True if no unsoundness was found
    """
    inner_lb, inner_ub = z_inner.concretize()  # Just for soundness checks
    outer_lb, outer_ub = z_outer.concretize()
    device = z_outer.head.device

    inner_lb -= z_inner.head
    inner_ub -= z_inner.head
    outer_lb -= z_outer.head
    outer_ub -= z_outer.head

    inner_lb, inner_ub, outer_lb, outer_ub = map(lambda x: x.flatten(start_dim=1), (inner_lb, inner_ub, outer_lb, outer_ub))

    # Get old and new error matrix
    if z_outer.domain == "chzono":
        if z_inner.beta is None:
            inner_errors = z_inner.get_errors(include_ch_box_term=True).flatten(start_dim=1)
        else:
            inner_errors = torch.cat([z_inner.get_errors(include_ch_box_term=True).flatten(start_dim=1), torch.diag(z_inner.beta.detach().flatten())], 0)
        if z_outer.beta is None:
            outer_errors = z_outer.get_errors(include_ch_box_term=True).flatten(start_dim=1)
        else:
            outer_errors = torch.cat([z_outer.get_errors(include_ch_box_term=True).flatten(start_dim=1), torch.diag(z_outer.beta.detach().flatten())], 0)
    else:
        if z_inner.box_errors is None or list(z_inner.box_errors.values()) == []:
            inner_errors = z_inner.get_zono_matrix().detach().flatten(start_dim=1)
        else:
            inner_errors = torch.cat([z_inner.get_zono_matrix().detach().flatten(start_dim=1),
                                      z_inner.get_box_matrix().detach().flatten(start_dim=1)], 0)
        if z_outer.box_errors is None or list(z_inner.box_errors.values()) == []:
            outer_errors = z_outer.get_zono_matrix().detach().flatten(start_dim=1)
        else:
            outer_errors = torch.cat(
                [z_outer.get_zono_matrix().detach().flatten(start_dim=1), z_outer.get_box_matrix().detach().flatten(start_dim=1)],
                0)

    outer_errors = outer_errors[~(outer_errors == 0).all(1)]
    inner_errors = inner_errors[~(inner_errors == 0).all(1)]

    # Sample random error terms from {-1,1} to make sure we get a corner of the zonotope.
    inner_error_terms = (1 - 2 * torch.randint(0, 2, (n, inner_errors.shape[0]))).to(DTYPE_TORCH).to(device)
    inner_points = torch.matmul(inner_error_terms, inner_errors)

    assert (outer_lb <= inner_points).all() and (inner_points <= outer_ub).all(), "No concrete containment"

    with torch.enable_grad():
        # initialize new error terms with a possible solution (not restricted to -1 < error terms < 1)
        outer_error_terms = torch.tensor(
            np.stack([np.linalg.lstsq(outer_errors.T.cpu().numpy(), inner_points[i].cpu().numpy(), rcond=None)[0] for i in range(n)], 0),
            dtype=DTYPE_TORCH, device=device).requires_grad_(True)
        outer_error_terms.data = torch.clip(outer_error_terms.data, -1, 1)
        num_steps = 10000
        opt = torch.optim.Adam([outer_error_terms], lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, 0.02 * num_steps, gamma=0.2)

        for _ in range(num_steps):
            outer_points = torch.matmul(outer_error_terms, outer_errors)

            loss = torch.square((outer_points - inner_points)).sum(1).sqrt()

            if (loss < 1e-8).all():
                #"solution found for all points => contain might be sound"
                break

            loss.sum().backward()
            opt.step()
            opt.zero_grad()
            outer_error_terms.data = torch.clip(outer_error_terms.data, -1, 1)
            lr_scheduler.step()

    # if verbose:
    #     print(f"Containment stats: {torch.max((outer_ub - outer_lb) / (inner_ub - inner_lb + 1e-8))} (concrete), mean abs error term {outer_error_terms.data.abs().mean()}")

    # if verbose and torch.max((outer_ub - outer_lb) / (inner_ub - inner_lb + 1e-8)) > 10:
    #     print(f"Maximum width increase of {torch.max((outer_ub - outer_lb) / (inner_ub - inner_lb + 1e-8))}")
    if verbose and not (loss < 1e-8).all():
        print(f"Containment soundness test failed with {loss[loss>1e-8]} ")
        contained_new, containment_factor = z_outer.contains(z_inner, "basis")  # TODO containment method
    return (loss < 1e-8).all()

class step_lr_scheduler:
    def __init__(self, initial_step_size, gamma=0.1, interval=10):
        self.initial_step_size = initial_step_size
        self.gamma = gamma
        self.interval = interval
        self.current_step = 0

    def step(self, k=1):
        self.current_step += k

    def get_lr(self):
        if isinstance(self.interval, int):
            return self.initial_step_size * self.gamma**(np.floor(self.current_step/self.interval))
        else:
            phase = len([x for x in self.interval if self.current_step>=x])
            return self.initial_step_size * self.gamma**(phase)


def check_soundness_of_transformation(z_old, z, use_sampling=False, require_contain_sound=True):
    old_lb, old_ub = z_old.concretize()
    new_lb, new_ub = z.concretize()

    # Checks if our transformation is contained
    contained_conc = ((new_lb <= old_lb) & (old_ub <= new_ub)).all()
    contained_sound, containment_factor = z.contains(z_old)  # This projects back into the old basis for containment check
    if use_sampling:
        with torch.enable_grad():
            contained_sampling = check_soundness_by_sampling(z_old, z)
    else:
        contained_sampling = "Unknown"

    if not (contained_conc and (contained_sound or not require_contain_sound)):
        print(f"Projection violates soundness!")
        print(f"Transform contained concrete: {contained_conc.all()} | Transform contained sound: {contained_sound} with {containment_factor} | Transform contained sampling {contained_sampling}")
        print(f"Min upper bound increase: {torch.min(new_ub - old_ub)}, Min lower bound decrease: {torch.max(new_lb - old_lb)}")
        assert False


def basis_transform(new_basis, old_vectors, ERROR_EPS_ADD=0, ERROR_EPS_MUL=0, return_full_res=False,
                    dtype=torch.float64):
    # We solve for the coordinates (x) of curr_basis (B) in new_basis (A)
    # I.e. we solve Ax=b
    A = new_basis
    B = old_vectors

    if A.shape[0] < 500 or A.shape[0] != A.shape[1]:
        # depending on the size of the matrices different methods are faster
        sol = np.linalg.lstsq(A, B, rcond=None)[0]
    else:
        sol = torch.solve(torch.tensor(B), torch.tensor(A)).solution.cpu().numpy()

    assert np.isclose(np.matmul(A, sol), B, atol=1e-7, rtol=1e-6).all(), f"Projection into new base errors failed"

    # We add the component ERROR_EPS_ADD to ensure the resulting error matrix has full rank and to compensate for potential numerical errors
    # Also this is how widening is implemented
    x = np.sum(np.abs(sol), axis=1) * (1 + ERROR_EPS_MUL) + ERROR_EPS_ADD

    res = torch.tensor(x.reshape(1, -1) * new_basis, dtype=dtype).T.unsqueeze(1)

    if return_full_res:  # sometimes we also want the see how the consolidation coefficients were obtained
        return res, sol
    else:
        return torch.tensor(x.reshape(1, -1) * new_basis, dtype=dtype).T.unsqueeze(1)


def get_new_basis(errors_to_get_basis_from, method="pca"):
    """
    Compute a bais of error directions from errors_to_get_basis_from
    :param errors_to_get_basis_from: Error matrix to be overapproximated
    :param method: "pca" or "pca_zero_mean"
    :return: a basis of error directions
    """
    if method == "pca":
        U, S, Vt = np.linalg.svd((errors_to_get_basis_from - errors_to_get_basis_from.mean(0)).cpu(), full_matrices=False)
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

    elif method == "pca_adaptive":
        # @Mark
        # Step 1 - PCA Basis
        pca_result = pca.fit(errors_to_get_basis_from.cpu())
        pca_basis_vectors = pca_result.components_.T

        # Step 2 - Remove basis vectors without magnitude
        rank = np.linalg.matrix_rank(errors_to_get_basis_from.cpu())

        # Should we use q = pca_result.singular_values_ > eps?
        reduced_pca = pca_basis_vectors[:, :rank]
        k = pca_basis_vectors.shape[1] - reduced_pca.shape[1]

        # Step 3 - Represent the old vectors in the new basis
        new_basis, full_coords = basis_transform(reduced_pca, errors_to_get_basis_from.T, return_full_res=True)
        # TODO check if orthonormal
        #  => orthonormalize if not

        # Step 4 - Select the top k vectors that are represented the worst
        # Can do many things here
        sums = torch.einsum("ij -> j", torch.abs(torch.Tensor(full_coords)))
        # TODO Do we need some rescaling here? I think the PCA terms are all normalized. This might be suboptimal
        # TODO also do we actually want the sum of the absolute terms (a perfectly captured/aligned but large term will score poorly here) or do we want to find a term with the largest increase in representation length
        #  => sums = torch.einsum("ij -> j", torch.abs(torch.Tensor(full_coords))) - errors_to_get_basis_from.square().sum(1).sqrt() # normalize with errors_to get basis from l2 norm # first term is sum of l2 norms
        top_k_indices = sums > torch.quantile(sums, 0.75)
        top_k_vectors = errors_to_get_basis_from.T[:, top_k_indices]

        # Step 5 - again use PCA to select the remaining vectors
        pca = PCA()
        pca_extension_vectors = pca.fit(top_k_vectors.T)

        # Step 6 - Final result
        new_basis_vectors = np.concatenate((reduced_pca, pca_extension_vectors.components_.T[:, :k]), axis=1)
        # TODO => this will not be full rank (most likely). Should we fix/change this?

        # Step 7 - What did we gain? We'll that's a good question!
        new_basis, new_coords = basis_transform(new_basis_vectors, errors_to_get_basis_from.T, return_full_res=True)

        new_sum = torch.einsum("ij -> j", torch.abs(torch.Tensor(new_coords)))

        print(f"Old sums {sums}")
        print(f"New sums {new_sum}")  # TODO Would max(np.abs(new_coords).sum(1)) (potentially after some rescaling) not be a better metric? There we get improvement
    elif method == "dict_learning":
        # NOTE: This performes badly
        # NOTE: X = U @ V
        # X = (n_errors_old, x_dim) The old error vectors are in column format
        # U = (n_errors_old, dim_basis_new) Contains the coordinates in the new basis in row format (unused) - Sparseness regularized by alpha
        # V = (dim_basis_new, x_dim) Contains the new basis in column format - Unit column norm
        # NOTE: The result is highly approximative for larger alpha due to regularization
        # NOTE: transform_algorithm{‘lasso_lars’, ‘lasso_cd’, ‘lars’, ‘omp’, ‘threshold’}, default=’omp’
        dict_learner = DictionaryLearning(n_components=100, transform_algorithm='omp', random_state=42, alpha=0,
                                          tol=1e-8)
        e_transformed = dict_learner.fit_transform(errors_to_get_basis_from)  # Coordinates in the new system row_wise
        new_basis_vectors = dict_learner.components_.T
        reconst = e_transformed @ dict_learner.components_
        avg_scaled_error = np.mean(np.sum((reconst - errors_to_get_basis_from.detach().numpy()) ** 2, axis=1) / np.sum(
            errors_to_get_basis_from.detach().numpy() ** 2, axis=1))
    else:
        assert False, f"Base computation method {method} is not recognized"

    if new_basis_vectors.shape[0] > new_basis_vectors.shape[1]:
        # not a full basis
        new_basis_vectors = complete_basis(new_basis_vectors)

    return new_basis_vectors


def gramm_schmidt_completion(A, r):
    n = A.shape[0]
    B = np.zeros((n, n))
    B[:, :r] = A[:, :r]
    j = r
    for i in range(n):
        if j == n:
            break
        # Set a as a unit vector e_i
        a = np.zeros_like(B[:, 0])
        a[i] = 1.
        for k in range(j):
            a = a - project(B[:, k], a)
        if np.abs(a).sum() < 1e-8:
            # included in linear span of selected vectors
            continue
        B[:, j] = a
        j += 1
    return B


def complete_basis(A):
    """
    Complete the full rank matrix A to a basis
    :param A: full rank matrix
    :return: A basis including the matrix A
    """
    n = A.shape[-2]
    n_a = A.shape[-1]
    B = np.zeros((n, n))
    B[:, :n_a] = A
    return gramm_schmidt_completion(A, n_a)


def project(u,v):
    return np.dot(u,v)/np.dot(u,u)*u

def plot_HCAS_pred(model, new_specLB, new_specUB):
    fig, ax = plt.subplots(figsize=(5, 4))

    x, y, z = [], [], []
    for lb, ub in zip(new_specLB, new_specUB):
        center = (lb + ub) / 2
        x.append(center[0])
        y.append(center[1])
        z.append(model(center).argmax(1))

    x = torch.stack(x).flatten().cpu()
    y = torch.stack(y).flatten().cpu()
    z = torch.stack(z).flatten().cpu()

    for label in range(5):
        if label == 0:  # COC clear of contact
            c = "black"
        elif label == 1:  # Weak Left
            c = "blue"
        elif label == 3:  # Strong Left
            c = "purple"
        elif label == 2:  # Weak Right
            c = "green"
        elif label == 4:  # Strong Right
            c = "gray"
        ax.scatter(x[z == label], y[z == label], color=c)
    plt.show()

def plot_HCAS_cert(cert_ranges):
    fig, ax = plt.subplots(figsize=(5, 4))
    for lb, ub, label in cert_ranges:
        lb, ub = lb.flatten(), ub.flatten()
        x = [lb[0], lb[0], ub[0], ub[0], ]
        y = [lb[1], ub[1], ub[1], lb[1], ]

        if label == 0:  # COC clear of contact
            c = "black"
        elif label == 1:  # Weak Left
            c = "blue"
        elif label == 3:  # Strong Left
            c = "purple"
        elif label == 2:  # Weak Right
            c = "green"
        elif label == 4:  # Strong Right
            c = "gray"
        ax.fill(x, y, c)
    plt.show()


def save_for_SemiSDP(model, save_path):
    state_dict = model.state_dict()
    state_dict = {re.sub("mon\.", "", k): v for k, v in state_dict.items()}
    state_dict = {re.sub("linear_module\.", "", k): v for k, v in state_dict.items()}
    state_dict = {re.sub("Wout\.", "C.", k): v for k, v in state_dict.items()}
    state_dict_final = {}
    for k, v in state_dict.items():
        k_letter = re.match("([a-z,A-z])\..*", k).group(1)
        k_letter = k_letter.lower() if "bias" in k else k_letter.upper()
        state_dict_final[k_letter] = v.detach().cpu().numpy()

    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"models",save_path)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    for k, v in state_dict_final.items():
        var_path = os.path.join(dir_path, f"{save_path}_{k}")
        np.save(var_path, v)


def containment_scale_binary_search(outer: HybridZonotope, inner_reference:HybridZonotope, scale_ub: float, eps: float, method: str, verbose: int) -> Tuple[float, float, HybridZonotope]:
    scale_lb = 1
    while scale_ub - scale_lb > eps:
        zu_n_scaled = inner_reference.clone()
        scale = 1 / 2. * (scale_ub + scale_lb)
        if verbose > 0:
            print(f"\nTrying containment with scaling {scale}, search range {scale_ub-scale_lb}>{eps}, Dim: {np.prod(inner_reference.shape)}, Error terms: {sum([x.shape[0] for x in inner_reference.zono_errors.values()])}/{sum([x.shape[0] for x in outer.zono_errors.values()])}, None beta: {inner_reference.beta is None}/{outer.beta is None}")
        zu_n_scaled = (zu_n_scaled - zu_n_scaled.head) * scale + zu_n_scaled.head
        if outer.contains(zu_n_scaled, method)[0]:
            scale_lb = scale
        else:
            scale_ub = scale
    zu_n_scaled = inner_reference.clone()
    zu_n_scaled = (zu_n_scaled - zu_n_scaled.head) * scale_lb + zu_n_scaled.head

    return scale_lb, scale_ub, zu_n_scaled