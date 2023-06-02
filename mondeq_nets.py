from ai.lbzonotope import LBZonotope
from ai.new_zonotope import HybridZonotope
from ai.deep_poly_forward import DeepPoly_f
from ai.abstract_layers import Sequential
from monDEQ.splitting import MONPeacemanRachford, MONForwardBackwardSplitting
import torch
import torch.nn as nn
import torch.nn.functional as F
import monDEQ.mon as mon
import numpy as np
from monDEQ.utils import apply_final_layer
from utils import expand_args, check_soundness_of_transformation, containment_scale_binary_search,check_soundness_by_sampling
import tqdm
from time import time

MON_DEFAULTS = {
    'alpha': 1.0,
    'tol': 1e-5,
    'max_iter': 300
}


class NormalizedNet(torch.nn.Module):
    def __init__(self, mean, sigma, net, input_dim):
        super(NormalizedNet, self).__init__()
        mean = torch.tensor(mean)
        sigma = torch.tensor(sigma)
        self.mean = nn.Parameter(mean.view((1, mean.numel(),) + (1,) *(len(input_dim)-1)), requires_grad=False)
        self.sigma = nn.Parameter(sigma.view((1, sigma.numel(),) + (1,) *(len(input_dim)-1)), requires_grad=False)
        self.net = net
        self.aWin = Sequential.from_concrete_network(net.Win if isinstance(net.Win, nn.Sequential) else nn.Sequential(net.Win), input_dim)
        self.aWout = Sequential.from_concrete_network(net.Wout if isinstance(net.Wout, nn.Sequential) else nn.Sequential(net.Wout), self.net.linear_module.out_dim)
        self.fwdbwd = MONForwardBackwardSplitting(net.linear_module, net.nonlin_module)
        self.pr = MONPeacemanRachford(net.linear_module, net.nonlin_module)

    def forward_zono(self, x, target, adaptive_a, iters, base_alpha, i_proj_start, dil_new_base, widening_iter,
                     fp_init=False, peaceman_rachford=False, verbose=False, dtype=torch.float32, check_sound=True,
                     post_cont_proj_dil=10, pre_cont_proj_dil=1, joint_projection=True, switch_to_fwbw=None,
                     log_file=None, optimize_slopes=False, require_contained=False, no_box_term=False, unsound_zono=False,
                     containment_experiment=False):

        zono_orig = x.clone()#HybridZonotope.construct_from_bounds(x.concretize()[0], x.concretize()[1], dtype=x.dtype, domain="zono")

        x = x.normalize(self.mean, self.sigma)
        self.net.linear_module.reset()
        self.aWout.reset_bounds()
        self.aWin.reset_bounds()

        domain = x.domain

        # Setup
        if isinstance(self.net, SingleFcNet):
            x = x.flatten()
        x = self.aWin(x)
        x_lb, x_ub = x.concretize()
        x_center = (x_ub + x_lb) / 2
        z_init = self.net.mon(x_center)[0]
        switched = False

        if verbose:
            print(f"Initial z {z_init}")

        if fp_init:
            # z = LBZonotope.lb_zonotope_from_noise(z_init, 0.0, error_id=1, dtype=dtype, data_min=None, data_max=None)
            z = z_init
        else:
            z = torch.zeros_like(z_init)

        if domain == "LBZonotope":
            z = LBZonotope(z, None, None)
        elif domain == "DPF":
            z = DeepPoly_f.construct_constant(z, x.inputs)
        else:
            z = HybridZonotope(z, None, None, domain)


        containing_lb, containing_ub = None, None
        contained = False
        f_lb = None
        lb, ub = -np.inf * torch.ones_like(z_init), np.inf * torch.ones_like(z_init)
        f_lb_best = -np.inf
        best_cont = np.inf
        best_cont_idx = -1
        contained_iter = -1
        projection_counter = 0

        if optimize_slopes >= 10:
            deepz_lambdas = torch.nn.parameter.Parameter(-torch.ones_like(z.concretize()[0]))
            lambda_opt = torch.optim.Adam([deepz_lambdas], lr=1e-2)
        else:
            deepz_lambdas = None

        alpha = base_alpha
        stable_range = iters

        past_zonotopes = []
        past_soundness = []
        k_step = 10
        steps_since_last_imp = 0

        # For unsound iterations we now can set the corresponding ERROR_EPS_ADD and ERROR_EPS_MULT here
        SOUND_ERROR_EPS_ADD = 1e-8
        SOUND_ERROR_EPS_MUL = 1e-8

        if widening_iter != 0:
            CURR_ERROR_EPS_ADD = 1e-2
            CURR_ERROR_EPS_MUL = 1e-3
        else:
            CURR_ERROR_EPS_ADD = SOUND_ERROR_EPS_ADD
            CURR_ERROR_EPS_MUL = SOUND_ERROR_EPS_MUL

        if peaceman_rachford:  # Initialize the inverse once
            # Expands to the correct V
            self.net.linear_module.init_inverse(1 + alpha, -alpha)
            if fp_init:
                u = z_init
            else:
                u = torch.zeros_like(z_init)

            if domain == "LBZonotope":
                u = LBZonotope(u, None, None)
            elif domain == "DPF":
                u = DeepPoly_f.construct_constant(u, x.inputs)
            else:
                u = HybridZonotope(u, None, None, domain)

            if len(x.shape)>2:
                Uxb = self.net.linear_module.cpad(x).conv2d(self.net.linear_module.U.weight, self.net.linear_module.U.bias,
                                                        self.net.linear_module.U.stride, self.net.linear_module.U.padding,
                                                        self.net.linear_module.U.dilation, self.net.linear_module.U.groups)
            else:
                Uxb = x.linear(self.net.linear_module.U.weight, self.net.linear_module.U.bias).detach()


        # Main verification loop, we start at some step with projecting onto the PCA-basis and do so every dilation steps
        #   We only keep the merge errors of the last iteration in the daigonal basis and move all others into the zono_error set (happens in the ReLU transformer)
        #   We project the zono errors (which also contain merge errors up to iteration i-1) onto the PCA Basis
        #   -> The merge errors from the previous iteration can now be captured with the PCA instead of diagonal basis
        #   -> (Not yet implemented) As this is not strictly better than V1, we can for containment checks move violating components from one base to the other and see if we can show containment that way

        # For each Zonotope set we keep additionally the sign vector so that we can flip signs accordingly in case of a negative correlation of error terms (this should be checked)
        pbar = tqdm.trange(stable_range)

        mean_widths = []
        pr_iter = 0
        fwdbwd_iter = 0
        pr_alpha = -1
        fwdbwd_alpha = -1
        n_cont_check = 0

        for i in pbar:
            IS_SOUND = CURR_ERROR_EPS_MUL >= SOUND_ERROR_EPS_MUL and CURR_ERROR_EPS_ADD >= SOUND_ERROR_EPS_ADD  # Make sure projections are sound
            if require_contained:
                contained = False


            ### The fixpoint iteration itself
            if peaceman_rachford:
                pr_iter += 1
                pr_alpha = alpha
                z_n, u_n = self.pr.forward_step(x, z, u, alpha, None, Uxb)
                if u_n.domain == "LBZonotope":
                    u_n.box_errors_to_zono()  # Moves old box_errors to zono_errors to match with z_n and allow for compensation.
            else:  # Forward-backward
                fwdbwd_iter += 1
                fwdbwd_alpha = alpha
                z_n = self.fwdbwd.forward_step(x, z, alpha, deepz_lambdas if contained else None)

            if no_box_term:
                if z_n.domain == "LBZonotope":
                    z_n.box_errors_to_zono()
                if z_n.domain == "chzono":
                    z_n.ch_zono_errors = None

            lb_new, ub_new = z_n.concretize()
            assert (lb_new <= ub_new).all()

            mean_widths.append((ub_new-lb_new).mean())

            ### Check for containment if not already found
            if not contained and len(past_zonotopes) > 0:
                if torch.max(ub_new - ub) <= 0 and torch.min(lb_new - lb) >= 0:
                    # box containment is given
                    # zu_n = (LBZonotope.cat([z_n, u_n], 1) if domain=="LBZonotope" else HybridZonotope.cat([z_n, u_n], 1)) if peaceman_rachford else z_n  # if doing pr check that both u and z are contained
                    zu_n = type(z_n).cat([z_n, u_n], 1) if peaceman_rachford else z_n  # if doing pr check that both u and z are contained
                    for j, past_z in enumerate(past_zonotopes):  # Iterate over the last k zonotopes to check for containment
                        # try:
                        if containment_experiment and zu_n.domain == "zono":
                            for scale in reversed([1.0, 0.9, 0.75, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005]):
                                print(f"\nTrying containment with scaling {scale}, Dim: {np.prod(zu_n.shape)}, Error terms: {sum([x.shape[0] for x in zu_n.zono_errors.values()])}/{sum([x.shape[0] for x in past_z.zono_errors.values()])}, None beta: {zu_n.beta is None}/{past_z.beta is None}")
                                z_test = zu_n.clone()
                                z_test = (z_test-z_test.head) * scale + z_test.head
                                cont, fact = past_z.contains(z_test, "gurobi")
                                if not cont:
                                    break
                        s = time()
                        contained_new, containment_factor = past_z.contains(zu_n, "basis_new") # TODO containment method
                        if contained_new and check_sound:
                            with torch.enable_grad():
                                assert check_soundness_by_sampling(zu_n, past_z, verbose=True)
                        if containment_experiment and contained_new and zu_n.domain == "chzono":
                            scale_lb, _, zu_n_scaled = containment_scale_binary_search(past_z, zu_n, scale_ub=1.2, eps=0.000001, method="basis", verbose=0)
                            print(f"Containment factor: {containment_factor}/{1/scale_lb}, time: {time()-s:.6f}")
                            scale_lb, scale_ub, _ = containment_scale_binary_search(past_z, zu_n_scaled, scale_ub=1.05, eps=0.002, method="gurobi", verbose=1)
                            # for scale in reversed([1.5, 1.2, 1.1, 1.05, 1.03, 1.024, 1.022, 1.02, 1.018, 1.016, 1.014, 1.012, 1.01, 1.0]):
                            #     print(f"\nTrying containment with scaling {scale}, Dim: {np.prod(zu_n.shape)}, Error terms: {sum([x.shape[0] for x in zu_n.zono_errors.values()])}/{sum([x.shape[0] for x in past_z.zono_errors.values()])}, None beta: {zu_n.beta is None}/{past_z.beta is None}")
                            #     z_test = zu_n_scaled.clone()
                            #     z_test = (z_test-z_test.head) * scale + z_test.head
                            #     cont, fact = past_z.contains(z_test, "gurobi")
                            #     if not cont:
                            #         scale_ub = scale
                            #         break
                            #     else:
                            #         scale_lb = scale
                            if log_file is not None:
                                print(f"Max containment scale lb: {scale_lb}; ub: {scale_ub}")
                                log_file.write(f"Max containment scale lb: {scale_lb}; ub: {scale_ub}\n")

                        n_cont_check += 1
                        # except:
                        #     contained_new = False
                        #     containment_factor = np.inf
                        if containment_factor < best_cont:
                            best_cont = containment_factor
                            best_cont_idx = j
                        if contained_new:
                            if all(past_soundness[j:]):
                                contained = True
                                contained_iter = i
                                if verbose:
                                    print(f"i: {i}, {j} Containment found")
                            else:
                                if verbose:
                                    print(f"i: {i}, Found containment using unsound iter")
                            widening_iter = 0  # Stop widening
                            CURR_ERROR_EPS_ADD = SOUND_ERROR_EPS_ADD
                            CURR_ERROR_EPS_MUL = SOUND_ERROR_EPS_MUL
                            if optimize_slopes >= 10:
                                lambda_opt.zero_grad()

                            break

            ### Check certifiability of classification
            if contained:
                # Containment only needs to be shown once
                f_lb, f_ub = apply_final_layer(z_n, self.aWout, target, verbose, pool=self.net.pool if hasattr(self.net,"pool") else None)

                if torch.min(f_lb) >= 0:  # Certification successful
                    if log_file is not None:
                        log_file.write(
                            f"Contained: {contained}, mean width {(ub - lb).mean():.5e}, best_cont: {best_cont:.2e} @ {best_cont_idx} in {contained_iter}/{n_cont_check}, min_lb: {f_lb.topk(dim=1, k=2, largest=False)[0].view(-1)[-1]:.6f} in {i + 1}\n")
                    if unsound_zono:
                        self.forward_mirror(zono_orig, target, pr_iter, pr_alpha, fwdbwd_iter, fwdbwd_alpha, 0, fp_init, log_file)

                    return True, (f_lb, f_ub)

                ### Optimize slopes
                if optimize_slopes >= 10 and not peaceman_rachford and not (deepz_lambdas < 0).all():

                    # loss = (torch.nn.functional.softplus(-f_lb*5 + 0.1)/5).sum()
                    loss = -1 * torch.min(f_lb)
                    loss.backward()
                    lambda_opt.step()
                    deepz_lambdas.data = torch.clamp(deepz_lambdas.data, 0, 1)
                    lambda_opt.zero_grad()
                    if (projection_counter + 2) % (post_cont_proj_dil) == 0:
                        deepz_lambdas.data = -torch.ones_like(deepz_lambdas.data)
                    z_n = z_n.detach()
                    if peaceman_rachford:
                        u_n = u_n.detach()

            ### Consolidate Error terms via Basis Transformation
            basis_transform_method = "pca_zero_mean"  # "pca_zero_mean, pca"
            with torch.no_grad():
                if i >= i_proj_start and \
                        ((not contained and (i - i_proj_start) % pre_cont_proj_dil == 0)
                         or (contained and (projection_counter + 1) % post_cont_proj_dil == 0)):
                    # Do projections
                    projected = True
                    if (not contained and (i - i_proj_start) % dil_new_base == 0) \
                            or (contained and projection_counter + 1 >= post_cont_proj_dil * 2):
                        # compute new basis
                        zu_n_basis = None
                        z_n_basis = None
                        u_n_basis = None
                        projection_counter = 0

                    if peaceman_rachford and joint_projection:
                        z_n_old = z_n
                        zu_n = type(z_n).cat([z_n, u_n], 1)
                        zu_n_old = zu_n

                        if domain in ["LBZonotope", "chzono"]:
                            try:
                                zu_n, zu_n_basis = zu_n.consolidate_errors(zu_n_basis, basis_transform_method,
                                                                       (verbose and i < 0),
                                                                       CURR_ERROR_EPS_ADD, CURR_ERROR_EPS_MUL)
                            except:
                                projected = False
                        elif domain == "DPF":
                            bias_delta = zu_n.x_u_bias - zu_n.x_l_bias
                            zu_n.x_l_bias = zu_n.x_l_bias - CURR_ERROR_EPS_MUL * bias_delta - CURR_ERROR_EPS_ADD
                            zu_n.x_u_bias = zu_n.x_u_bias + CURR_ERROR_EPS_MUL * bias_delta + CURR_ERROR_EPS_ADD
                        elif domain == "box":
                            zu_n.beta = zu_n.beta * (1+CURR_ERROR_EPS_MUL) + CURR_ERROR_EPS_ADD
                        elif domain == "zono":
                            zu_n = zu_n.widen(CURR_ERROR_EPS_ADD, CURR_ERROR_EPS_MUL)
                        z_n = zu_n[:, :z_n.shape[1]]
                        u_n = zu_n[:, z_n.shape[1]:]
                    else:
                        z_n_old = z_n
                        if domain in ["LBZonotope", "chzono"]:
                            try:
                                z_n, z_n_basis = z_n.consolidate_errors(z_n_basis, basis_transform_method, (verbose and i < 0),
                                                            CURR_ERROR_EPS_ADD, CURR_ERROR_EPS_MUL)
                            except:
                                projected = False

                        elif domain == "DPF":
                            bias_delta = z_n.x_u_bias - z_n.x_l_bias
                            z_n.x_l_bias = z_n.x_l_bias - CURR_ERROR_EPS_MUL * bias_delta - CURR_ERROR_EPS_ADD
                            z_n.x_u_bias = z_n.x_u_bias + CURR_ERROR_EPS_MUL * bias_delta + CURR_ERROR_EPS_ADD
                        elif domain == "box":
                            z_n.beta = z_n.beta * (1+CURR_ERROR_EPS_MUL) + CURR_ERROR_EPS_ADD
                        elif domain == "zono":
                            z_n = z_n.widen(CURR_ERROR_EPS_ADD, CURR_ERROR_EPS_MUL)
                        if peaceman_rachford:
                            u_n_old = u_n
                            if domain in  ["LBZonotope", "chzono"]:
                                try:
                                    u_n, u_n_basis = u_n.consolidate_errors(u_n_basis, basis_transform_method,
                                                                    (verbose and i < 0), CURR_ERROR_EPS_ADD,
                                                                    CURR_ERROR_EPS_MUL)
                                except:
                                    projected = False

                            elif domain == "DPF":
                                bias_delta = u_n.x_u_bias - u_n.x_l_bias
                                u_n.x_l_bias = u_n.x_l_bias - CURR_ERROR_EPS_MUL * bias_delta - CURR_ERROR_EPS_ADD
                                u_n.x_u_bias = u_n.x_u_bias + CURR_ERROR_EPS_MUL * bias_delta + CURR_ERROR_EPS_ADD
                            elif domain == "box":
                                u_n.beta = u_n.beta * (1 + CURR_ERROR_EPS_MUL) + CURR_ERROR_EPS_ADD
                    lb_new, ub_new = z_n.concretize()
                    if IS_SOUND and check_sound and domain in ["LBZonotope", "chzono"]:
                        if peaceman_rachford:
                            if joint_projection:
                                check_soundness_of_transformation(zu_n_old, zu_n, use_sampling=False,
                                                              require_contain_sound=True)
                            else:
                                check_soundness_of_transformation(u_n_old, u_n, use_sampling=False,
                                                              require_contain_sound=True)
                                check_soundness_of_transformation(z_n_old, z_n,
                                                                  use_sampling=False,
                                                                  require_contain_sound=True)
                        else:
                            check_soundness_of_transformation(z_n_old, z_n,
                                                              use_sampling=False,
                                                              require_contain_sound=True)
                else:
                    projected = False

            if containing_lb is not None and (
                    torch.max(ub_new - containing_ub) > 0 or torch.min(lb_new - containing_lb) < 0):
                print(
                    f"Violation of containing bounds i: {i}, max upper bound dim {torch.argmax(ub_new - containing_ub)} increace: {torch.max(ub_new - containing_ub)}, max lower bound dim {torch.argmin(lb_new - containing_lb)} decrease: {torch.min(lb_new - containing_lb)}")

            # Update at end of iteration
            if projected or domain=="box":
                # Containment checks only work well if if the larger zonotope has a Basis of d terms
                if peaceman_rachford and (not joint_projection or domain != "LBZonotope"):
                        zu_n = type(z_n).cat([z_n, u_n], 1)
                past_zonotopes.append(z_n if not peaceman_rachford else zu_n)
                past_soundness.append(IS_SOUND)
                if len(past_zonotopes) > k_step:
                    past_zonotopes = past_zonotopes[-k_step:]
                    past_soundness = past_soundness[-k_step:]

            # Set parameters for next iteration
            if contained:
                projection_counter += 1
                if projection_counter >= 20 and not switched:
                    switched = True
                    if switch_to_fwbw is not None and peaceman_rachford:
                        peaceman_rachford = False
                        alpha = switch_to_fwbw
                        # del zu_n, u_n, u, u_n_basis, zu_n_basis
                        if adaptive_a:
                            with torch.no_grad():
                                alpha = compute_adaptive_alpha(z_n, z, x, self.fwdbwd, self.aWout,
                                                               target, alpha, method="line-search", pool=self.net.pool if hasattr(self.net,"pool") else None)
                            if log_file is not None:
                                log_file.write(f"Alpha computed: {alpha:.5f}\n")
                                # print(f"Alpha computed: {alpha:.5f}")

                        # For PR the concatenated zonotopes are compared
                        if joint_projection:
                            past_zonotopes = [zu_n[:, :z_n.shape[1]] for zu_n in past_zonotopes]
                    elif adaptive_a:
                        pass
                        # with torch.no_grad():
                        #     alpha = compute_adaptive_alpha(zu_n, z, x, self.pr, self.aWout,
                        #                                   target, alpha, method="line-search",
                        #                                   pool=self.net.pool if hasattr(self.net, "pool") else None)
                        #     self.net.linear_module.Winv = None
                        #     self.net.linear_module.init_inverse(1 + alpha, -alpha)
                        #     Uxb = None
                        # if log_file is not None:
                        #     log_file.write(f"Alpha computed: {alpha:.5f}\n")
                            # print(f"Alpha computed: {alpha:.5f}")

            elif widening_iter > 0 and i % widening_iter == 0:
                # exponentially increasing widening
                CURR_ERROR_EPS_ADD *= 1.2
                CURR_ERROR_EPS_MUL *= 1.1

            if contained and IS_SOUND:
                if f_lb is not None:
                    if f_lb.min() > f_lb_best:
                        f_lb_best = f_lb.min()
                        steps_since_last_imp = 0
                    else:
                        steps_since_last_imp += 1

            if steps_since_last_imp >= 3 * post_cont_proj_dil and projection_counter + 2 == post_cont_proj_dil:
                break

            z = z_n
            if peaceman_rachford:
                u = u_n
            lb = lb_new
            ub = ub_new

            min_lb_ver = "None" if f_lb is None else f"{f_lb.min():.4f}"
            pbar.set_description(f"Contained: {contained}, mean width {(ub - lb).mean():.5e}, alpha: {alpha:.5f}, best_cont_res: {best_cont-1:.2e} @ {best_cont_idx} in {contained_iter}/{n_cont_check}, min_lb_ver: {min_lb_ver}, min_lb_best: {f_lb_best:.4f}")
            if (ub - lb).max() > 1e9:
                break

        if contained and f_lb_best > -1 and optimize_slopes % 10 >= 1:# and not peaceman_rachford:
            # first do cheap unrolling
            optimize_slopes_used = 10 * (optimize_slopes // 10) + 1
            grad_verified, (f_lb, f_ub), target_idxs = grad_optimize_via_unroll(zu_n if peaceman_rachford else z_n, z, x, self.pr if peaceman_rachford else self.fwdbwd,
                                                                                self.aWout,
                                                                                target, alpha,
                                                                                pool=self.net.pool if hasattr(self.net,"pool") else None)
            if torch.min(f_lb) > -0.15 and not grad_verified and optimize_slopes % 10 >= 2:
                # do longer unrolles, more optimizations and against specific labels
                optimize_slopes_used = 10 * (optimize_slopes // 10) + 2
                grad_verified, (f_lb, f_ub), target_idxs = grad_optimize_via_unroll(zu_n if peaceman_rachford else z_n, z, x,  self.pr if peaceman_rachford else self.fwdbwd,
                                                                                    self.aWout,
                                                                                    target, alpha, unroll=40,
                                                                                    optim_steps=200, lr=1e-2,
                                                                                    target_idx=target_idxs,
                                                                                    pool=self.net.pool if hasattr(self.net,"pool") else None)

            if f_lb.min() == 0:
                f_lb_best = f_lb.topk(dim=1, k=2, largest=False)[0].view(-1)[-1]
            else:
                f_lb_best = max(f_lb_best, f_lb.min())
            if log_file is not None:
                log_file.write(f"Contained: {contained}, mean width {(ub - lb).mean():.5e}, best_cont: {best_cont:.2e} @ {best_cont_idx} in {contained_iter}/{n_cont_check}, min_lb: {f_lb_best:.6f} in {i + 1}g\n")
                if unsound_zono:
                    self.forward_mirror(zono_orig, target, pr_iter, pr_alpha, fwdbwd_iter, fwdbwd_alpha, optimize_slopes_used, fp_init, log_file)

            return grad_verified, (f_lb, f_ub)

        if not contained:
            f_lb, f_ub = None, None

        if log_file is not None:
            log_file.write(f"Contained: {contained}, mean width {(ub - lb).mean():.5e}, best_cont: {best_cont:.2e} @ {best_cont_idx} in {contained_iter}/{n_cont_check}, min_lb: {f_lb_best:.6f} in {i + 1}\n")
            if unsound_zono:
                self.forward_mirror(zono_orig, target, pr_iter, pr_alpha, fwdbwd_iter, fwdbwd_alpha, 0, fp_init, log_file)

        return False, (f_lb, f_ub)

        # out = self.net.forward_ver_zonotope(x, target, adaptive_a, iters, base_alpha, i_proj_start, dil_new_base,
        #                                     widening_iter=widening_iter, fp_init=fp_init,
        #                                     peaceman_rachford=peaceman_rachford, verbose=verbose, dtype=dtype,
        #                                     check_sound=check_sound,
        #                                     post_cont_proj_dil=post_cont_proj_dil, pre_cont_proj_dil=pre_cont_proj_dil,
        #                                     joint_projection=joint_projection, switch_to_fwbw=switch_to_fwbw, log_file=log_file,
        #                                     optimize_slopes=optimize_slopes)
        # return out

    def forward(self, x, *args, **kwargs):
        if isinstance(x, torch.Tensor):
            x = (x - self.mean) / self.sigma
        else:
            x = x.normalize(self.mean, self.sigma)
        return self.net(x, *args, **kwargs)

    def forward_mirror(self, x, target, pr_iter, pr_alpha, fwdbwd_iter, fwdbwd_alpha, optimize_slopes, fp_init, log_file):
        start = time()
        x = x.normalize(self.mean, self.sigma)
        self.net.linear_module.reset()
        self.aWout.reset_bounds()
        self.aWin.reset_bounds()

        domain = x.domain

        # Setup
        if isinstance(self.net, SingleFcNet):
            x = x.flatten()
        x = self.aWin(x)
        x_lb, x_ub = x.concretize()
        x_center = (x_ub + x_lb) / 2
        z_init = self.net.mon(x_center)[0]

        if fp_init:
            z = z_init
        else:
            z = torch.zeros_like(z_init)

        if domain == "LBZonotope":
            z = LBZonotope(z, None, None)
        elif domain == "DPF":
            z = DeepPoly_f.construct_constant(z, x.inputs)
        else:
            z = HybridZonotope(z, None, None, domain)

        peaceman_rachford = pr_iter > 0

        alpha = pr_alpha if peaceman_rachford else fwdbwd_alpha

        if peaceman_rachford:  # Initialize the inverse once
            self.net.linear_module.init_inverse(1 + alpha, -alpha)
            if fp_init:
                u = z_init
            else:
                u = torch.zeros_like(z_init)

            if domain == "LBZonotope":
                u = LBZonotope(u, None, None)
            elif domain == "DPF":
                u = DeepPoly_f.construct_constant(u, x.inputs)
            else:
                u = HybridZonotope(u, None, None, domain)

            if len(x.shape)>2:
                Uxb = self.net.linear_module.cpad(x).conv2d(self.net.linear_module.U.weight, self.net.linear_module.U.bias,
                                                        self.net.linear_module.U.stride, self.net.linear_module.U.padding,
                                                        self.net.linear_module.U.dilation, self.net.linear_module.U.groups)
            else:
                Uxb = x.linear(self.net.linear_module.U.weight, self.net.linear_module.U.bias).detach()

        for i in range(pr_iter+fwdbwd_iter):
            if i == pr_iter:
                peaceman_rachford = False
                alpha = fwdbwd_alpha
            if peaceman_rachford:
                z, u = self.pr.forward_step(x, z, u, alpha, None, Uxb)
                if u.domain in ["LBZonotope", "chzono"]:
                    u.box_errors_to_zono()  # Moves old box_errors to zono_errors to match with z_n and allow for compensation.
                zu = type(z).cat([z, u], 1)
            else:  # Forward-backward
                z = self.fwdbwd.forward_step(x, z, alpha, None)

            lb, ub = z.concretize()

        f_lb, f_ub = apply_final_layer(z, self.aWout, target, 0, pool=self.net.pool if hasattr(self.net, "pool") else None)

        if optimize_slopes % 10 >= 1:
            # first do cheap unrolling
            grad_verified, (f_lb, f_ub), target_idxs = grad_optimize_via_unroll(zu if peaceman_rachford else z, z, x, self.pr if peaceman_rachford else self.fwdbwd,
                                                                                self.aWout,
                                                                                target, alpha,
                                                                                pool=self.net.pool if hasattr(self.net,"pool") else None)
            if optimize_slopes % 10 == 2:
                # do longer unrolles, more optimizations and against specific labels
                grad_verified, (f_lb, f_ub), target_idxs = grad_optimize_via_unroll(zu if peaceman_rachford else z, z, x, self.pr if peaceman_rachford else self.fwdbwd,
                                                                                    self.aWout,
                                                                                    target, alpha, unroll=40,
                                                                                    optim_steps=200, lr=1e-2,
                                                                                    target_idx=target_idxs,
                                                                                    pool=self.net.pool if hasattr(self.net,"pool") else None)

        if f_lb.min() == 0:
            f_lb_best = f_lb.topk(dim=1, k=2, largest=False)[0].view(-1)[-1]
        else:
            f_lb_best = f_lb.min()
        if log_file is not None:
            log_file.write(f"UNSOUND: mean width {(ub - lb).mean():.5e}, min_lb: {f_lb_best:.6f}, time: {time()-start:.6f}\n")
        print(f"UNSOUND: mean width {(ub - lb).mean():.2e}, min_lb: {f_lb_best:.4f}\n")


class SingleFcNet(nn.Module):
    def __init__(self, splittingMethod, in_dim=784, latent_dim=100, alpha=1.0, max_iter=300, tol=1e-5, m=0.1, Win=None, Wout=None, n_class=10, **kwargs):
        super().__init__()
        # self.alpha = alpha          # NOTE Not used atm
        # self.max_iter = max_iter
        # self.tol = tol
        if Win is None:
            Win = nn.Identity()
        if Wout is None:
            Wout = nn.Linear(latent_dim, n_class)
        self.Win = Win
        self.out_dim = latent_dim
        self.linear_module = mon.MONSingleFc(in_dim, latent_dim, m=m)
        self.nonlin_module = mon.MONReLU()
        self.mon = splittingMethod(self.linear_module, self.nonlin_module, alpha=alpha, tol=tol, max_iter=max_iter, **expand_args(kwargs))
        self.Wout = Wout

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.Win(x)
        z = self.mon(x)
        # z_n = self.mon.nonlin_module(self.mon.linear_module(x, *z)[0])

        return self.Wout(z[-1])


class SingleConvNet(nn.Module):

    def __init__(self, splittingMethod, in_dim=28, in_channels=1, out_channels=32, m=0.1, kernel_x=6, stride_x=3, alpha=1.0, max_iter=300, tol=1e-5, Win=None, Wout=None, n_class=10, **kwargs):
        super().__init__()
        n = in_dim #+ 2
        shp = (n, n)
        self.pool = 2
        if Win is None:
            Win = nn.Identity()
        self.Win = Win
        _, _, out_dim = getShapeConv((in_channels, in_dim, in_dim), (out_channels, kernel_x, kernel_x),
                                    stride=stride_x, padding=1)
        self.out_dim = out_channels * (out_dim // self.pool) ** 2
        self.linear_module = mon.MONSingleConv(in_channels, out_channels, shp, kernel_size=3, m=m, stride=stride_x, kernel_x=kernel_x)
        self.nonlin_module = mon.MONBorderReLU(self.linear_module.pad[0])
        self.mon = splittingMethod(self.linear_module, self.nonlin_module, alpha=alpha, max_iter=max_iter, tol=tol,**expand_args(kwargs))
        if Wout is None:
            Wout = nn.Linear(self.out_dim, n_class)
        self.Wout = Wout

    def forward(self, x):
        # x = F.pad(x, (1, 1, 1, 1))
        x = self.Win(x)
        z = self.mon(x)
        z = F.avg_pool2d(z[-1], self.pool)
        return self.Wout(z.view(z.shape[0], -1))


class ConvNet(nn.Module):

    def __init__(self, splittingMethod, in_dim=28, in_channels=1, out_channels=32, m=0.1, alpha=1.0, max_iter=300, tol=1e-5, Wout=None, lben=True, lben_cond=3,**kwargs):
        super().__init__()
        n = in_dim #+ 2
        shp = (n, n)
        self.pool = 4
        _, _, out_dim = getShapeConv((in_channels, in_dim, in_dim), (out_channels, 3, 3),
                                    stride=1, padding=(3 - 1)//2)
        self.out_dim = out_channels * (out_dim // self.pool) ** 2
        self.linear_module = mon.MONConv(in_channels, out_channels, shp, kernel_size=3, m=m, lben=lben, lben_cond=lben_cond)
        self.Win = nn.Identity() #self.linear_module.bias_layers
        self.nonlin_module = mon.MONReLU()
        self.mon = splittingMethod(self.linear_module, self.nonlin_module, alpha=alpha, max_iter=max_iter, tol=tol,**expand_args(kwargs))
        if Wout is None:
            Wout = [nn.BatchNorm2d(out_channels)]
            Wout += [nn.AvgPool2d(self.pool)]
            Wout += [nn.Flatten()]
            Wout += [nn.Linear(self.out_dim, 10)]
            Wout = nn.Sequential(*Wout)
        self.Wout = Wout

    def forward(self, x):
        # x = F.pad(x, (1, 1, 1, 1))
        x = self.Win(x)
        z = self.mon(x)[0]
        # z = F.avg_pool2d(z[-1], self.pool)
        return self.Wout(z)


class MultiConvNet(nn.Module):
    def __init__(self, splittingMethod, in_dim=28, in_channels=1, conv_sizes=(16, 32, 64), m=1.0, **kwargs):
        super().__init__()
        self.linear_module = mon.MONMultiConv(in_channels, conv_sizes, in_dim + 2, kernel_size=3, m=m)
        self.nonlin_module = mon.MONBorderReLU(self.linear_module.pad[0])
        self.mon = splittingMethod(self.linear_module, self.nonlin_module, **expand_args(MON_DEFAULTS, kwargs))
        out_shape = self.linear_module.z_shape(1)[-1]
        dim = out_shape[1] * out_shape[2] * out_shape[3]
        self.Wout = nn.Linear(dim, 10)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1))
        zs = self.mon(x)
        z = zs[-1]
        z = z.view(z.shape[0], -1)
        return self.Wout(z)


def gramm_schmidt(A, r):
    n = A.shape[0]
    B = np.zeros((n, n))
    B[:, :r] = A
    j = r
    for i in range(n):
        if j == n:
            print(f"done at i={i}")
            break
        a = np.zeros_like(B[:, 0])
        a[i] = 1.
        for k in range(j):
            a = a - project(B[:, k], a)
        if np.abs(a).sum() < 1e-8:
            continue
        B[:, j] = a
        j += 1
        print(B)
    return B


def lin_indep(A):
    indep = []
    n = A.shape[0]
    for j in range(n):
        a = A[:, j]
        for k in range(j):
            a -= project(A[:, k], a)
        if np.abs(a).sum() > 1e-8:
            print(np.abs(a).sum())
            print(a)
            indep.append(j)
    return indep


def project(u, v):
    return np.dot(u, v) / np.dot(u, u) * u


def compute_adaptive_alpha(z_n, z, x, mon, Wout, target, alpha, method="line-search", deepz_lambdas=None, pool=None):
        if method == "min_diag":
            pass
        #     if z_n.box_errors is None or z.box_errors is None:
        #         return alpha
        #     # Heuristic: Select alpha such that the over all diagonal sum is minimized
        #     d_old = torch.sum(torch.diagonal(z.box_errors))
        #     d_new = torch.sum(torch.diagonal(z_n.box_errors))
        #     if d_old * d_new < 0:
        #         alpha = (- d_old / (d_new - d_old)).item()
        #     print(alpha)
        #     assert alpha > 0 and alpha < 1, "alpha no longer in range"
        #     return alpha
        # elif method == "min_sum":
        #     # Heuristic: Select alpha such that the overall resulting error sum is minimized when sampling the respective space
        #     alpha_range = [1 / x for x in range(20, 49)]
        #
        #     best_alpha = alpha
        #     best_avg_width = 1e10
        #     # best_merge_sum = 1e10
        #     # best_merge_max = 1e10
        #     # best_zono_sum = 1e10
        #     # best_zono_max = 1e10
        #
        #     for test_alpha in alpha_range:
        #         z_test = (1 - test_alpha) * z + test_alpha * (z_n)
        #         # Values of potential interest
        #
        #         lb, ub = z_test.concretize()
        #         # merge_sum = torch.sum(torch.abs(z_test.box_errors))
        #         # merge_max = torch.max(torch.abs(z_test.box_errors))
        #         # zono_sum = torch.sum(torch.abs(z_test.get_zono_matrix()))
        #         # zono_max = torch.max(torch.abs(z_test.get_zono_matrix()))
        #
        #         avg_width = torch.mean(ub - lb)
        #
        #         if best_avg_width > avg_width:
        #             best_avg_width = avg_width
        #             best_alpha = test_alpha
        #
        #     return best_alpha
        elif method == "line-search":
            # To be optimized (though fast enough in practice) 
            alpha_options = [1/i for i in range(10, 41)] + [1/(42+i*2) for i in range(10)] #+ \
                            # [1/(65+i*5) for i in range(8)] + [1/(120+i*20) for i in range(5)] +\
                            # [1/(250+i*50) for i in range(6)] #+ \
                            # [1 / (600 + i * 100) for i in range(5)]+ \
                            # [1 / (1200 + i * 200) for i in range(5)]
            best_lb = np.NINF
            best_alpha = None
            lbs = []
            for alpha in alpha_options:
                z_t = z_n.clone()
                if isinstance(mon, MONPeacemanRachford):
                    # alpha = 2*alpha
                    mon.linear_module.Winv = None
                    mon.linear_module.init_inverse(1 + alpha, -alpha)
                    d_latent = int(z_t.shape[-1] / 2)
                    z_t, u_t = z_t[:, :d_latent], z_t[:, d_latent:]
                for i in range(10):
                    # z_n_t = mon.linear_module._forward_abs(x, z_t)
                    # z_n_t = (1 - alpha) * z_t + alpha * (z_n_t)
                    # z_t = mon.nonlin_module._forward_abs(z_n_t)
                    if isinstance(mon,MONPeacemanRachford):
                        z_t, u_t = mon.forward_step(x, z_t, u_t, alpha, deepz_lambdas)
                        if u_t.domain in ["LBZonotope", "chzono"]:
                            u_t.box_errors_to_zono()
                    else:
                        z_t = mon.forward_step(x, z_t, alpha, deepz_lambdas)

                f_lb, f_ub = apply_final_layer(z_t, Wout, target, False, pool)
                f_lb_min = f_lb.min()
                lbs.append(f_lb_min)
                if f_lb_min > best_lb:
                    best_alpha = alpha
                    best_lb = f_lb_min
                    if f_lb_min >= 0:
                        break
            else:
                if best_alpha is None:
                    best_alpha = alpha
            if isinstance(mon, MONPeacemanRachford):
                mon.linear_module.Winv = None
                mon.linear_module.init_inverse(1 + alpha, -alpha)
            return best_alpha
        # elif method == "gradient":
        #     with torch.enable_grad():
        #         # To be optimized (though fast enough in practice)
        #         deep_alpha = torch.nn.parameter.Parameter(torch.ones(1, device="cuda")*alpha)
        #
        #         optim_steps = 50
        #         unroll = 10
        #
        #         opt = torch.optim.Adam( [ {'params': [deep_alpha], 'lr':2e-4}])
        #         #scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
        #         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=10)
        #
        #         for i in range(optim_steps):
        #             z_t = z_n
        #             # Get result
        #             for j in range(unroll):
        #                 #deep_alpha = torch.clamp(deep_alpha, 1/20, 1/64)
        #                 z_n_t = mon.linear_module._forward_abs(x, z_t)
        #                 z_n_t = (1 - deep_alpha) * z_t + deep_alpha * (z_n_t)
        #                 z_t = mon.nonlin_module._forward_abs(z_n_t, None)
        #
        #             f_lb, f_ub = apply_final_layer(z_t, Wout, target, False, pool)
        #             loss = -1*torch.min(f_lb)
        #             loss.backward()
        #             opt.step()
        #             opt.zero_grad()
        #             scheduler.step(loss)
        #
        #             if torch.min(f_lb) >= 0:
        #                 break
        #
        #         return deep_alpha.detach().item()


def grad_optimize_via_unroll(z_n, z, x, iteration, Wout, label, alpha, unroll=20, optim_steps=60, lr=2e-2, target_idx=None, pool=None):
    with torch.enable_grad():
        target_idx = [None] if target_idx is None else target_idx
        f_lb_best = -np.inf
        
        non_ver_idx = set(range(Wout.layers[-1].weight.shape[0]))

        for target in target_idx:
            deepz_lambdas = [torch.nn.parameter.Parameter(-torch.ones_like(z.head, device=x.device, requires_grad=True)) for i in range(unroll)]
            # Setup deep lambda
            if isinstance(iteration, MONPeacemanRachford):
                # deep_alpha = [torch.ones(1, device=x.device) * alpha]
                opt = torch.optim.Adam([{'params': deepz_lambdas, 'lr': lr}])
            else:
                deep_alpha = [torch.nn.parameter.Parameter(torch.ones(1, device=x.device) * alpha) for i in range(unroll)]
                opt = torch.optim.Adam([{'params': deepz_lambdas, 'lr': lr},
                                    {'params': deep_alpha[0], 'lr': 2e-4}])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=10)

            pbar = tqdm.trange(optim_steps)
            if target is not None and target not in non_ver_idx:
                continue

            for i in pbar:
                z_t = z_n
                if isinstance(iteration, MONPeacemanRachford):
                    iteration.linear_module.Winv = None
                    d_latent = int(z_t.shape[-1] / 2)
                    z_t, u_t = z_t[:, :d_latent], z_t[:, d_latent:]
                    iteration.linear_module.init_inverse(1 + alpha, -alpha)
                    # assert deep_alpha>0
                # Get result
                for j in range(unroll):
                    if isinstance(iteration, MONPeacemanRachford):
                        z_t, u_t = iteration.forward_step(x, z_t, u_t, alpha, deepz_lambdas[j])
                        if u_t.domain in ["LBZonotope"]:
                            u_t.box_errors_to_zono()
                    else:
                        z_t = iteration.forward_step(x, z_t, deep_alpha[j], deepz_lambdas[j])

                f_lb, f_ub = apply_final_layer(z_t, Wout, label, False, pool)

                f_lb_best = max(f_lb_best, f_lb.min())
                non_ver_idx -= set([i for i in range(f_lb.shape[-1]) if f_lb[0][i] >= 0])
                if target is None:
                    pbar.set_description(f"Slope Optim {i}, min_lb: {f_lb.min():.4f}, min_lb_best: {f_lb_best:.4f}, non_verified:{non_ver_idx}")
                else:
                    pbar.set_description(f"Slope Optim {i} against {target}, lb_{target}: {f_lb[0][target]:.4f}, min_lb_best: {f_lb_best:.4f}, non_verified:{non_ver_idx}")

                if f_lb_best >= 0 or (target is not None and target not in non_ver_idx):
                    break

                if target is None:
                    loss = -1*torch.min(f_lb)
                else:
                    loss = -1 * f_lb[0][target]

                loss.backward()
                opt.step()
                for dl in deepz_lambdas:
                    dl.data = torch.clamp(dl.data, 0, 1)
                    assert (dl>=0).all() and (dl<=1).all()
                if not isinstance(iteration, MONPeacemanRachford):
                    for a in deep_alpha:
                        a.data = torch.clamp(a.data, min=0)
                        assert (a >= 0).all()
                opt.zero_grad()
                scheduler.step(loss)
            else:
                break # for loop was not aborted => target could not be certified

        return len(non_ver_idx) == 0, (f_lb, f_ub), non_ver_idx

def getShapeConv(in_shape, conv_shape, stride = 1, padding = 0):
    inChan, inH, inW = in_shape
    outChan, kH, kW = conv_shape[:3]

    outH = 1 + int((2 * padding + inH - kH) / stride)
    outW = 1 + int((2 * padding + inW - kW) / stride)
    return (outChan, outH, outW)