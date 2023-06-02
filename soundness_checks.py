import torch
import numpy as np
import random
import tqdm
import matplotlib.pyplot as plt
from typing import Optional
from utils import cuda

def check_bounds(model, data, bounds, eps, num_checks=1000, adv=False, seed=None, domain_lb:Optional[float]=0, domain_ub:Optional[float]=1) -> bool:
    # This can be batched for random checks -> I left it individual to easily move it to adv. attacks
    ex_data, ex_target = data[0], data[1]
    # bounds = (bounds[0].detach().numpy(), bounds[1].detach().numpy())

    if seed is None:
        seed = np.random.randint(0, 10000)

    if adv:
        specLB =  torch.maximum(ex_data - eps, torch.ones_like(ex_data)*domain_lb)
        specUB = torch.minimum(ex_data + eps, torch.ones_like(ex_data)*domain_ub)
        adv_robust, adex = _pgd_whitebox(model, ex_data, bounds, ex_target, specLB, specUB , ex_data.device,
                  num_steps=30, step_size=0.1, ODI_num_steps=5, ODI_step_size=1., restarts=1, seed=seed)
        violation_found = not adv_robust
    else:
        for i in tqdm.trange(num_checks):
            curr_data = torch.minimum(torch.maximum(ex_data + (torch.rand_like(ex_data)-0.5)*2*eps, torch.zeros_like(ex_data)), torch.ones_like(ex_data))
            preds = model(curr_data)
            target = ex_target.numpy()
            preds = preds.detach().numpy()

            sub_mat = -1 * np.eye(10)
            sub_mat[:, target] = 1
            sub_mat[target, :] = 0
            deltas = np.matmul(preds, sub_mat.T)
            assert (bounds[0] <= deltas).all() and (bounds[1] >= deltas).all()
            if not (bounds[0] <= deltas).all() and (bounds[1] >= deltas).all():
                violation_found = True
                break

    return violation_found


def check_robustness(model, batch, eps, seed=None, domain_lb=0, domain_ub=1):
    data, target = batch[0], batch[1]

    specLB = torch.maximum(data - eps, torch.ones_like(data)*domain_lb)
    specUB = torch.minimum(data + eps, torch.ones_like(data)*domain_ub)

    return _pgd_whitebox(model, data, None, target, specLB, specUB, data.device, num_steps=50, step_size=0.05,
                         ODI_num_steps=5, ODI_step_size=1., restarts=1, seed=seed, mode="accuracy")


def check_robustness_bounds(model, target, specLB, specUB, seed=None):
    return _pgd_whitebox(model, (specLB+specUB)/2., None, target, specLB, specUB, specLB.device, num_steps=50, step_size=0.05,
                         ODI_num_steps=5, ODI_step_size=1., restarts=1, seed=seed, mode="accuracy")

def _pgd_whitebox(model, X, bounds, target, specLB, specUB, device, num_steps=200, step_size=0.2,
                  ODI_num_steps=10, ODI_step_size=1., restarts=1, seed=None, mode="soundness", n_class=10):
    n_class = model(X).shape[-1]
    repeats = int(np.floor(100/2/n_class))
    batch_size = int(repeats*n_class*2)
    device = X.device
    dtype = X.dtype
    assert mode in ["soundness", "accuracy"]

    if seed is None:
        seed = np.random.randint(0, 10000)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    for _ in range(restarts):
        X_pgd = torch.autograd.Variable(X.data.repeat((batch_size,) + (1,) * (X.dim() - 1)), requires_grad=True).to(device)
        randVector_ = torch.ones_like(model(X_pgd)).uniform_(-1, 1)
        random_noise = torch.ones_like(X_pgd).uniform_(-0.5, 0.5)*(specUB-specLB)
        X_pgd = torch.autograd.Variable(torch.minimum(torch.maximum(X_pgd.data + random_noise, specLB), specUB), requires_grad=True)

        lr_scale = (specUB-specLB)

        for i in tqdm.trange(ODI_num_steps + num_steps+1):
            opt = torch.optim.SGD([X_pgd], lr=1e-1)
            opt.zero_grad()
            assert (X_pgd <= specUB).all() and (X_pgd >= specLB).all(), f"Adv example invalid"

            with torch.enable_grad():
                out = model(X_pgd)

                sub_mat = -1 * torch.eye(n_class, dtype=out.dtype, device=out.device)
                sub_mat[:, target] = 1
                sub_mat[target, :] = 0
                deltas = torch.matmul(out, sub_mat.T)

                if mode == "soundness":
                    if not ((bounds[0] <= deltas).all() and (bounds[1] >= deltas).all()):
                        violating_index = (bounds[0] > deltas.detach()).__or__(bounds[1] < deltas.detach()).sum(1).nonzero()[0][0]
                        print("Violating example:")
                        print(X_pgd[violating_index])
                        print("Instance bounds:")
                        print(specLB)
                        print(specUB)
                        print("Violating Output")
                        print(deltas[violating_index])
                        print("Output Bounds")
                        print(bounds[0])
                        print(bounds[1])
                        # plt.imshow(X_pgd[0].detach().cpu().numpy().transpose(1, 2, 0))
                        # plt.show()
                        return False, X_pgd[violating_index]
                    assert (bounds[0] <= deltas).all() and (bounds[1] >= deltas).all(), f"max lb violation: {torch.max(bounds[0] - deltas)}, max ub violation {torch.max(deltas - bounds[1])}"
                elif mode == "accuracy":
                    if not out.argmax(1).eq(target).all():
                        violating_index = (~out.argmax(1).eq(target)).nonzero()[0][0]
                        return False, X_pgd[violating_index]

                if i < ODI_num_steps:
                    loss = (out * randVector_).sum()
                else:
                    if mode == "soundness":
                        loss = torch.cat([torch.ones(repeats*n_class, dtype=dtype, device=device), -torch.ones(repeats*n_class, dtype=dtype, device=device)],0)*\
                           deltas[torch.eye(n_class, dtype=bool, device=device).repeat(2*repeats, 1)]
                    elif mode == "accuracy":
                        loss = -torch.ones(repeats*n_class*2, dtype=dtype, device=device) *\
                           deltas[torch.eye(n_class, dtype=bool, device=device).repeat(2*repeats, 1)]

            loss.sum().backward()
            if i < ODI_num_steps:
                eta = ODI_step_size * lr_scale * X_pgd.grad.data.sign()
            else:
                eta = lr_scale * step_size * X_pgd.grad.data.sign()
            X_pgd = torch.autograd.Variable(torch.minimum(torch.maximum(X_pgd.data + eta, specLB), specUB), requires_grad=True)
    # if mode == "accuracy":
    return True, None
