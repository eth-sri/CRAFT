import pprint
import time
import warnings
import os
import sys
import re
import matplotlib.pyplot as plt
import pickle

from ai.lbzonotope import LBZonotope
from ai.new_zonotope import HybridZonotope
from ai.deep_poly_forward import DeepPoly_f

import torch
import numpy as np
import scipy.io as sio
from typing import Tuple, List, Optional
from torch import Tensor

# import sys
# sys.path.append("monotone_op_net")
from monDEQ.train import mnist_loaders, cifar_loaders
from monDEQ.utils import HCAS_loader
import monDEQ.splitting as sp

from soundness_checks import check_bounds, check_robustness, check_robustness_bounds
from parse_args import parse_args
from utils import set_seeds, plot_HCAS_cert, plot_HCAS_pred, save_for_SemiSDP
from mondeq_nets import SingleFcNet, SingleConvNet, NormalizedNet, ConvNet
from bunch import Bunch  # type: ignore[import]


EPS = 1e-4
DTYPE = np.float64
DTYPE_TORCH = torch.float32 if DTYPE == np.float32 else torch.float64


seed = np.random.randint(0, 1000)
seed = 944
print(f"random seed: {seed}")

# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
set_seeds(seed)

def verify_for_eps(model, data, eps, max_iters, base_alpha, adaptive_a, i_proj_start, dil_new_base, widening_iter,
                   fp_init, peaceman_rachford, verbose, dtype, check_sound, post_cont_proj_dil=10, pre_cont_proj_dil=1,
                   joint_projection=True, switch_to_fwbw=None, log_file=None, optimize_slopes=False, domain="LBZonotope",
                   require_contained=False, no_box_term=False, unsound_zono=False, containment_experiment=False) -> bool:
    data, target = data[0], data[1]
    if domain == "LBZonotope":
        zono = LBZonotope.lb_zonotope_from_noise(data, eps, error_id=0, dtype=dtype)
    elif domain == "DPF":
        zono = DeepPoly_f.construct_from_noise(data, eps, dtype=dtype, data_range=(0, 1))
    else:
        zono = HybridZonotope.construct_from_noise(data, eps, domain=domain, dtype=dtype)


    res, bounds = model.forward_zono(zono, target.item(), 
                                    adaptive_a, max_iters, base_alpha, 
                                    i_proj_start, dil_new_base, widening_iter,
                                    fp_init, peaceman_rachford, verbose, dtype, check_sound=check_sound,
                                    post_cont_proj_dil=post_cont_proj_dil, pre_cont_proj_dil=pre_cont_proj_dil,
                                    joint_projection=joint_projection, switch_to_fwbw=switch_to_fwbw, log_file=log_file,
                                    optimize_slopes=optimize_slopes, require_contained=require_contained, no_box_term=no_box_term,
                                     unsound_zono=unsound_zono, containment_experiment=containment_experiment)
    return res, bounds





def verify_for_range(model, specLB, specUB, eps_max, max_iters, base_alpha, adaptive_a, i_proj_start, dil_new_base, widening_iter,
                   fp_init, peaceman_rachford, verbose, dtype, check_sound, post_cont_proj_dil=10, pre_cont_proj_dil=1,
                   joint_projection=True, switch_to_fwbw=None, log_file=None, optimize_slopes=False, domain="LBZonotope",
                   require_contained=False, no_box_term=False, remaining_depth=None, total_volume=None, certified_volume=0.,
                   adv_attack=False, level_zero=True) -> Tuple[List[Tuple[Tensor,Tensor,int]], float]:

    specLB, specUB = specLB.flatten(), specUB.flatten()
    specWidth = (specUB-specLB)
    if total_volume is None:
        total_volume = np.prod(specWidth.cpu().numpy())
    min_segments = torch.ceil(specWidth/eps_max).to(int)
    # min_segments[-1] = 1
    new_specLB, new_specUB = [], []
    cert_ranges = []

    for i in range(min_segments[0]):
        for j in range(min_segments[1]):
            for k in range(min_segments[2]):
                new_specLB.append(torch.stack([i / min_segments[0] * specWidth[0] + specLB[0],
                                   j / min_segments[1] * specWidth[1] + specLB[1],
                                   k / min_segments[2] * specWidth[2] + specLB[2],
                                   ]))
                new_specUB.append(torch.stack([(i+1) / min_segments[0] * specWidth[0] + specLB[0],
                                   (j+1) / min_segments[1] * specWidth[1] + specLB[1],
                                   (k+1) / min_segments[2] * specWidth[2] + specLB[2],
                                   ]))

    if level_zero:
        plot_HCAS_pred(model, new_specLB, new_specUB)

    for lb, ub in zip (new_specLB, new_specUB):
        start = time.time()
        lb, ub = lb.view(1, -1), ub.view(1, -1)
        center = (lb + ub) / 2
        volume = np.prod((ub-lb).flatten().cpu().numpy())/total_volume

        if domain == "LBZonotope":
            zono = LBZonotope.lb_zonotope_from_bounds(lb, ub, dtype=dtype)
        elif domain == "DPF":
            zono = DeepPoly_f.construct_from_bounds(lb, ub, dtype=dtype)
        else:
            zono = HybridZonotope.construct_from_bounds(lb, ub, domain=domain, dtype=dtype)

        assert torch.isclose(lb, zono.concretize()[0]).all() and torch.isclose(ub, zono.concretize()[1]).all()


        center_pred = model(center).argmax(1)
        # try:
        res, bounds = model.forward_zono(zono, center_pred.item(),
                                    adaptive_a, max_iters, base_alpha,
                                    i_proj_start, dil_new_base, widening_iter,
                                    fp_init, peaceman_rachford, verbose, dtype, check_sound=check_sound,
                                    post_cont_proj_dil=post_cont_proj_dil, pre_cont_proj_dil=pre_cont_proj_dil,
                                    joint_projection=joint_projection, switch_to_fwbw=switch_to_fwbw, log_file=log_file,
                                    optimize_slopes=optimize_slopes, require_contained=require_contained, no_box_term=no_box_term)

        if adv_attack:
            adv_res, adex = check_robustness_bounds(model, center_pred.item(), lb, ub, seed)
        else:
            adv_res = False

        if res:
            cert_ranges.append((lb, ub, center_pred))
            certified_volume += volume
            if adv_attack:
                assert adv_res, "Certified but counterexample found"

        print(f"Cert Volume: {certified_volume:.3e}, Range Volume: {volume: .3e}, Range: {lb.cpu().detach().numpy()}-{ub.cpu().detach().numpy()}, label: {center_pred.item()}, Converged: {bounds[0] is not None}, Verified: {res}, Adv Robust: {adv_res}")
        if log_file is not None:
            log_file.write(f"Cert Volume: {certified_volume:.3e}, Range Volume: {volume: .3e}, Range: {lb}-{ub}, label: {center_pred.item()}, Converged: {bounds[0] is not None}, Verified: {res}, Adv Robust: {adv_res}, time: {time.time() - start:.3f}\n\n")
        # except Exception as e:
        #     print("EXCEPTION encountered:")
        #     print(e)
        #     res = False
        #     if log_file is not None:
        #         log_file.write("EXCEPTION encountered")
        #         log_file.write(str(e))



        if not res and remaining_depth > 0:
            cert_ranges_new, certified_volume = verify_for_range(model, lb, ub, eps_max/2, max_iters, base_alpha, adaptive_a, i_proj_start,
                             dil_new_base, widening_iter,
                             fp_init, peaceman_rachford, verbose, dtype, check_sound, post_cont_proj_dil,
                             pre_cont_proj_dil, joint_projection, switch_to_fwbw, log_file, optimize_slopes,
                             domain, require_contained, no_box_term, remaining_depth-1, total_volume, certified_volume,
                                                                 level_zero=False)
            cert_ranges += cert_ranges_new

    if level_zero:
        plot_HCAS_cert(cert_ranges)

    return cert_ranges, certified_volume


def verify(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not isinstance(args, Bunch):
        args = Bunch(vars(args))

    m = args.m
    h = args.hidden
    num_verifications = args.number_verifications
    model_path = args.path
    adaptive_a  = args.adaptive_alpha
    iters       = args.iters
    base_alpha  = args.alpha
    eps         = args.eps
    example     = args.explicit
    i_proj_start= args.proj_start
    dil_new_base= args.dilation_new_base
    widening_iter = args.widening_iter
    fp_init     = args.fixed_point_init
    pr_iteration= args.peaceman_rachford
    verbose     = args.verbose
    dtype = torch.float64 if args.double else torch.float32
    check_sound = args.check_sound
    adv_attack = args.adv_attack
    post_cont_proj_dil = args.post_proj_dil
    pre_cont_proj_dil = args.pre_proj_dil
    joint_projection = args.joint_projection
    switch_to_fwbw = None if args.switch_to_fwbw == False else args.switch_to_fwbw
    log = args.log
    optimize_slopes = args.optimize_slopes
    domain = args.domain
    dataset = args.dataset
    k = args.get("kernel_x", None)
    s = args.get("stride_x", None)
    require_contained = args.get("require_contained",False)
    no_box_term = args.get("no_box_term", False)
    verify_range = args.get("verify_range", False)
    range_depth = args.get("range_depth", 0.)
    range_x = torch.tensor(args.get("range_x", [0., 0.]), dtype=dtype, device=device)
    range_y = torch.tensor(args.get("range_y", [0., 0.]), dtype=dtype, device=device)
    range_psi = torch.tensor(args.get("range_psi", [0., 0.]), dtype=dtype, device=device)
    unsound_zono = args.get("unsound_zono", False)
    containment_experiment = args.get("containment_experiment", False)

    # device = "cpu"

    if log is not None:
        if not os.path.isdir('logs'):
            os.mkdir('logs')
        try:
            log_file = open(log, "w")
            if isinstance(args, Bunch):
                pprint.pprint(dict(args.items()), log_file)
            else:
                pprint.pprint(vars(args), log_file)
            log_file.write(f"Random seed: {seed}\n")
            log_file.write(f"Using device: {device}\n\n")
            log_file.write(f"\nPython Version:\n{sys.version}")
            log_file.write(f"\nTorch Version:\n{torch.__version__}")
            log_file.write(f"CUDA Version:\n{torch.version.cuda}")
            log_file.write("\nDevice Info:")
            n_device = torch.cuda.device_count()
            for i in range(n_device):
                log_file.write(f"{i}: {torch.cuda.get_device_name(i)}")


        except:
            warnings.warn(f"Could not create log file: {log}")
            log_file = None
    else:
        log_file = None

    if isinstance(args, Bunch):
        pprint.pprint(dict(args.items()))
    else:
        pprint.pprint(vars(args))
    print(f"Using device {device}")
    print(f"\nPython Version:\n{sys.version}")
    print(f"\nTorch Version:\n{torch.__version__}")
    print(f"CUDA Version:\n{torch.version.cuda}")
    print("\nDevice Info:")
    n_device = torch.cuda.device_count()
    for i in range(n_device):
        print(f"{i}: {torch.cuda.get_device_name(i)}")

    # Instantiate and load monDEQ model
    if model_path.endswith(".mat"):
        model_type = "mat_models"
    elif os.path.basename(model_path).startswith("IBP"):
        model_type = "ibp_models"
    else:
        model_type = "models"

    if dataset == "mnist":
        mean = [0.1307,]
        std = [0.3081,]
        shp_d = [1, 28, 28]
        domain_lb, domain_ub = 0,1
        trainLoader, testLoader, _, _ = mnist_loaders(train_batch_size=128, test_batch_size=1, normalize=False, augment=False)
        n_class = 10
    elif dataset == "cifar":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        shp_d = [3, 32, 32]
        domain_lb, domain_ub = 0, 1
        trainLoader, testLoader, _, _ = cifar_loaders(train_batch_size=128, test_batch_size=1, normalize=False, augment=False)
        n_class = 10
    elif dataset.startswith("HCAS"):
        match = re.match("HCAS_p([0-9])_t([0-9]*)", dataset)
        if match is None:
            pra = 0
            tau = 10
        else:
            pra = int(match.group(1))
            tau = int(match.group(2))

        x_bounds = [-56000, 56000]
        y_bounds = [-56000, 56000]
        psi_bounds = [-180, 180]
        x_width = x_bounds[1] - x_bounds[0]
        y_width = y_bounds[1] - y_bounds[0]
        psi_width = psi_bounds[1] - psi_bounds[0]

        shp_d = [3]
        domain_lb, domain_ub = 0, 1
        trainLoader, testLoader, mean, std = HCAS_loader(pra=pra, tau=tau, train_batch_size=128, test_batch_size=1, normalize=False)
        n_class = 5
    else:
        assert False, f"Dataset {dataset} is unknown"

    if model_type == "models":
        if args.conv:
            model = SingleConvNet(sp.MONPeacemanRachford, in_dim=shp_d[-1],
                          in_channels=shp_d[0],
                          out_channels=h,
                          m=m, kernel_x=k, stride_x=s,
                          tol=1e-6, max_iter=300, n_class=n_class)
            # model = SingleConvNet(sp.MONForwardBackwardSplitting, in_dim=28,
            #               in_channels=1,
            #               out_channels=h,
            #               m=m, alpha=0.033)
        else:
            model = SingleFcNet(sp.MONPeacemanRachford, in_dim=np.prod(shp_d), latent_dim=h, alpha=0.2, max_iter=300,
                                tol=1e-7, m=m, n_class=n_class)
            # model = SingleFcNet(sp.MONForwardBackwardSplitting, in_dim=np.prod(shp_d), latent_dim=h, alpha=0.033, max_iter=300,
            #                     tol=1e-7, m=m, n_class=n_class)
        unexpected, extra = model.load_state_dict(torch.load(model_path, map_location=torch.device(device)), strict=False)
        assert len(extra) == 0
    elif model_type == "mat_models":
        # model = SingleFcNet(sp.MONForwardBackwardSplitting, in_dim=np.prod(shp_d), latent_dim=87, alpha=0.033, max_iter=300,
        #                     tol=1e-7, m=m, n_class=n_class)
        model = SingleFcNet(sp.MONPeacemanRachford, in_dim=np.prod(shp_d), latent_dim=87, alpha=0.2, max_iter=300,
                            tol=1e-7, m=m, n_class=n_class)
        mat_contents = sio.loadmat(model_path)
        # Mapping
        # U = U 
        model.mon.linear_module.U.weight = torch.nn.Parameter(torch.Tensor(mat_contents['U']))
        model.mon.linear_module.U.bias   = torch.nn.Parameter(torch.Tensor(mat_contents['u']).flatten())
        # A = A
        model.mon.linear_module.A.weight = torch.nn.Parameter(torch.Tensor(mat_contents['A']))
        # B = B 
        model.mon.linear_module.B.weight = torch.nn.Parameter(torch.Tensor(mat_contents['B']))
        # W = C
        model.Wout.weight = torch.nn.Parameter(torch.Tensor(mat_contents['C']))
        model.Wout.bias   = torch.nn.Parameter(torch.Tensor(mat_contents['c']).flatten())
        print("Loaded Matlab model")
    elif model_type == "ibp_models":
        model = ConvNet(sp.MONForwardBackwardSplitting, in_dim=shp_d[-1],
                          in_channels=shp_d[0], out_channels=h,
                          m=m, lben=args.lben, lben_cond=args.lben_cond, tol=1e-6, max_iter=300, n_class=n_class)
        state_dict_loaded = torch.load(model_path, map_location=torch.device(device))[0]
        state_dict_loaded = {re.sub("layers\.layers\.0\.", "", k): v for k, v in state_dict_loaded.items()}
        state_dict_loaded = {re.sub("Wout\.", "Wout.3.", k): v for k, v in state_dict_loaded.items()}
        state_dict_loaded = {re.sub("layers\.layers\.1\.", "Wout.0.", k): v for k, v in state_dict_loaded.items()}
        state_dict_loaded = {re.sub("U\.M\.", "U.", k): v for k, v in state_dict_loaded.items()}

        missing, extra = model.load_state_dict(state_dict_loaded, strict=False)
        assert len(extra) == 0
        model.linear_module.prep_model()
    else:
        assert False, f"Model type {model_type} not recognized"

    if isinstance(model, SingleFcNet):
        save_for_SemiSDP(model, re.sub("\.pt","",args.path.split("/")[-1]))
        W = model.mon.linear_module.get_W()
        a_max = 2 * m / np.linalg.norm((torch.eye(len(W)) - W).detach().cpu().numpy(), ord=2)**2
        # model.mon.alpha = 0.95 * a_max
        print(f"alpha max for FwdBwd is {a_max:.8f}")
        if log_file is not None:
            log_file.write(
                f"FwdBwd alpha max: {a_max:.8f}\n\n")

    model = model.to(device).to(dtype)
    model.eval()
    model_normalized = NormalizedNet(mean, std, model, input_dim=shp_d).to(dtype).to(device)
    model_normalized.eval()


    if verify_range:
        # specLB_old = torch.tensor([-5000/112000+0.5, -10000/112000+0.5, -90.5/360+0.5], device=device, dtype=dtype)
        # specUB_old = torch.tensor([20000/112000+0.5, 15000/112000+0.5, -89.5/360+0.5], device=device, dtype=dtype)
        specLB = torch.tensor([(range_x[0]-x_bounds[0]) / x_width, (range_y[0]-y_bounds[0]) / y_width, (range_psi[0]-psi_bounds[0]) / psi_width], device=device,
                              dtype=dtype)
        specUB = torch.tensor([(range_x[1]-x_bounds[0]) / x_width, (range_y[1]-y_bounds[0]) / y_width, (range_psi[1]-psi_bounds[0]) / psi_width], device=device,
                              dtype=dtype)
        start = time.time()

        certified_ranges, certified_volume = verify_for_range(model_normalized, specLB, specUB, eps_max=eps, max_iters=iters,
                       base_alpha=base_alpha, adaptive_a=adaptive_a, i_proj_start=i_proj_start,
                       dil_new_base=dil_new_base, widening_iter=widening_iter, fp_init=fp_init,
                       verbose=verbose, peaceman_rachford=pr_iteration, dtype=dtype, check_sound=check_sound,
                       post_cont_proj_dil=post_cont_proj_dil, pre_cont_proj_dil=pre_cont_proj_dil,
                       joint_projection=joint_projection, switch_to_fwbw=switch_to_fwbw, log_file=log_file,
                       optimize_slopes=optimize_slopes, domain=domain,
                       require_contained=require_contained, no_box_term=no_box_term, adv_attack=adv_attack,
                                                              remaining_depth=range_depth)

        end = time.time()

        print(f"Certified Volume: {certified_volume:.4f}, Time: {end-start:.2f}")
        if log is not None:
            log_file.write(f"Certified Volume: {certified_volume:.4f}, Time: {end-start:.2f}")
            dump_file = log[:-4] + "_ranges.pkl"
            certified_ranges = [[y.flatten().cpu().detach().numpy() for y in x] for x in certified_ranges]
            pickle.dump(certified_ranges, open(dump_file, "wb"))

    elif example > -1: # Test a specific example with the given index
        data = ((testLoader.dataset.test_data[example].reshape((1, 1, 28, 28)).to(dtype).to(device) / 255),
                testLoader.dataset.test_labels[example].to(device))
        set_seeds(seed)

        verif_res, bounds = verify_for_eps(model_normalized, data, eps=eps, max_iters=iters,
                                           base_alpha=base_alpha, adaptive_a=adaptive_a, i_proj_start=i_proj_start,
                                           dil_new_base=dil_new_base, widening_iter=widening_iter, fp_init=fp_init,
                                           verbose=verbose, peaceman_rachford=pr_iteration, dtype=dtype, check_sound=check_sound,
                                           post_cont_proj_dil=post_cont_proj_dil, pre_cont_proj_dil=pre_cont_proj_dil,
                                           joint_projection=joint_projection, switch_to_fwbw=switch_to_fwbw, log_file=log_file,
                                           optimize_slopes=optimize_slopes, domain=domain,
                                           require_contained=require_contained, no_box_term=no_box_term)

        print(f"Image: {example+1}: Converged: {bounds[0] is not None}, Verified: {verif_res}")

        if check_sound and bounds[0] is not None:
            check_bounds(model_normalized, data, bounds, eps, num_checks=100, adv=True, seed=seed)

    else:   # Test num verification images
        counter_nat = 0
        counter_ver = 0
        counter_adv = 0
        counter_conv = 0
        time_all = 0.0  # Time over all samples
        time_nat = 0.0  # Time only over samples where unperturbed sample is classified correctly (Reported)
        time_adv = 0.0  # Time only over samples where no adversarial example was found

        for i, batch in enumerate(testLoader):
            if i >= num_verifications: break

            print(f"Currently at image: {i+1}/{num_verifications}")
            start = time.time()
            if dataset == "mnist":
                batch = (testLoader.dataset.data[i].reshape((1, *shp_d)).to(dtype).to(device) / 255.,
                         testLoader.dataset.targets[i].to(device))

            elif dataset == "HCAS":
                # batch = (testLoader.dataset.tensors[0][i].reshape((1, *shp_d)).to(dtype).to(device),
                #          testLoader.dataset.tensors[1][i].to(device))
                batch = (batch[0].to(dtype).to(device),batch[1].to(device),)
                pass
            else:
                batch = ((torch.tensor(testLoader.dataset.data[i].transpose(2, 0, 1).reshape((1, *shp_d)), dtype=dtype, device=device) / 255.),
                      torch.tensor(testLoader.dataset.targets[i], device=device))

            set_seeds(seed)

            # Is the original sample classified correctly?
            x, y = batch
            if not (y.dim() <= 1 or y.shape[1] == 1):
                batch = (x, torch.argmax(y))

            nat_out = model_normalized(x)
            if y.dim() == 0 or y.shape[-1] == 1:
                nat_res = torch.argmax(nat_out, 1).eq(y).item()
            else:
                nat_res = torch.argmax(nat_out, 1).eq(torch.argmax(y,-1)).item()
            if nat_res:
                counter_nat += 1

            if optimize_slopes >= 10:
                # For optimize slope mode >= 10, we need gradients
                with torch.enable_grad():
                    verif_res, bounds = verify_for_eps(model_normalized, batch, eps=eps, max_iters=iters, base_alpha=base_alpha,
                                                   adaptive_a=adaptive_a, i_proj_start=i_proj_start, dil_new_base=dil_new_base,
                                                   widening_iter=widening_iter, fp_init=fp_init, peaceman_rachford=pr_iteration,
                                                   verbose=verbose, dtype=dtype, check_sound=check_sound,
                                                   post_cont_proj_dil=post_cont_proj_dil, pre_cont_proj_dil=pre_cont_proj_dil,
                                                   joint_projection=joint_projection, switch_to_fwbw=switch_to_fwbw, log_file=log_file,
                                                   optimize_slopes=optimize_slopes, domain=domain,
                                                   require_contained=require_contained, no_box_term=no_box_term,
                                                   unsound_zono=unsound_zono, containment_experiment=containment_experiment)
            else:
                with torch.no_grad():
                    verif_res, bounds = verify_for_eps(model_normalized, batch, eps=eps, max_iters=iters, base_alpha=base_alpha,
                                                    adaptive_a=adaptive_a, i_proj_start=i_proj_start, dil_new_base=dil_new_base,
                                                    widening_iter=widening_iter, fp_init=fp_init, peaceman_rachford=pr_iteration,
                                                    verbose=verbose, dtype=dtype, check_sound=check_sound,
                                                    post_cont_proj_dil=post_cont_proj_dil, pre_cont_proj_dil=pre_cont_proj_dil,
                                                    joint_projection=joint_projection, switch_to_fwbw=switch_to_fwbw, log_file=log_file,
                                                    optimize_slopes=optimize_slopes, domain=domain,
                                                    require_contained=require_contained, no_box_term=no_box_term,
                                                    unsound_zono=unsound_zono, containment_experiment=containment_experiment)

            end = time.time()
            time_all += end - start
            if nat_res:
                time_nat += end - start

            # Can we find an adversarial example?
            if adv_attack:
                adv_res, adex = check_robustness(model_normalized, batch, eps, seed, domain_lb, domain_ub)
                if adv_res:
                    time_adv += end - start

                if verif_res:
                    assert adv_res, "Certified but not empirically robust"
            else:
                adv_res = 0

            if check_sound and bounds[0] is not None:
                # Check via sampling if the obtained bounds on the classification margin can be broken via adversarial search
                sampling_succ = check_bounds(model_normalized, batch, bounds, eps, num_checks=100, adv=True, seed=seed)
                assert not sampling_succ, "Bound violation found"

            if verif_res:
                counter_ver += 1
            if adv_res:
                counter_adv += 1
            if bounds[0] is not None:
                counter_conv += 1

            print(f"Image: {i+1}: Converged: {bounds[0] is not None} [{counter_conv}/{i+1}], Verified: {verif_res} [{counter_ver}/{counter_conv}], Adv Robust: {adv_res} [{counter_adv}/{i+1}], Nat: {nat_res} [{counter_nat}/{i+1}]")
            if log_file is not None:
                log_file.write(f"Image: {i+1}: Converged: {bounds[0] is not None} [{counter_conv}/{i+1}], Verified: {verif_res} [{counter_ver}/{counter_conv}], Adv Robust: {adv_res} [{counter_adv}/{i+1}], Nat: {nat_res} [{counter_nat}/{i+1}], time: {end-start:.6f}/{time_all / (i + 1):.3f}/{(0 if counter_nat == 0 else time_nat / counter_nat):.3f}/{(0 if counter_adv == 0 else time_adv / counter_adv):.3f}/{time_all:.3f}\n\n")

        print(f"Converged: [{counter_conv}/{i }], Verified: [{counter_ver}/{counter_conv}], Adv Robust: [{counter_adv}/{i}], Nat: [{counter_nat}/{i}]")

        if log_file is not None:
            log_file.close()
        return 0

if __name__ == "__main__":
    args = parse_args()
    verify(args)

