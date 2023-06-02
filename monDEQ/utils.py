import torch
import numpy as np
from typing import Optional, Union, Tuple
import h5py
from torch.utils.data import DataLoader, TensorDataset

"""
Based on: https://github.com/locuslab/monotone_op_net/blob/master/utils.py
"""

class Meter(object):
    """Computes and stores the min, max, avg, and current values"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -float("inf")
        self.min = float("inf")

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)
        self.min = min(self.min, val)


class SplittingMethodStats(object):
    def __init__(self):
        self.fwd_iters = Meter()
        self.bkwd_iters = Meter()
        self.fwd_time = Meter()
        self.bkwd_time = Meter()

    def reset(self):
        self.fwd_iters.reset()
        self.fwd_time.reset()
        self.bkwd_iters.reset()
        self.bkwd_time.reset()

    def report(self):
        print('Fwd iters: {:.2f}\tFwd Time: {:.4f}\tBkwd Iters: {:.2f}\tBkwd Time: {:.4f}\n'.format(
                self.fwd_iters.avg, self.fwd_time.avg,
                self.bkwd_iters.avg, self.bkwd_time.avg))


def cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def apply_final_layer(z, Wout, target, verbose, pool=None):
    containing_lb, containing_ub = z.concretize()
    containing_lb, containing_ub = containing_lb.detach().clone(), containing_ub.detach().clone()
    if verbose:
        print(containing_lb)
        print(containing_ub)

    if len(z.shape)>2:
        if pool is not None:
            z = z.avg_pool2d(pool, pool)
        z = z.flatten()

    final_zonotope = Wout(z)

    n_class = final_zonotope.shape[-1]

    sub_mat = -1*torch.eye(n_class, dtype=final_zonotope.head.dtype, device=z.head.device)
    sub_mat[:, target] = 1
    sub_mat[target, :] = 0
    final_zonotope = final_zonotope.matmul(sub_mat.T)

    f_lb, f_ub = final_zonotope.concretize()
    return f_lb, f_ub


def compute_eigval(lin_module, method="power", compute_smallest=False, largest=None):
    with torch.no_grad():
        if method == "direct":
            W = lin_module.W.weight
            eigvals = torch.symeig(W + W.T)[0]
            return eigvals.detach().cpu().numpy()[-1] / 2

        elif method == "power":
            z0 = tuple(torch.randn(*shp).to(lin_module.U.weight.device) for shp in lin_module.z_shape(1))
            lam = power_iteration(lin_module, z0, 100,
                                  compute_smallest=compute_smallest,
                                  largest=largest)
            return lam


def power_iteration(linear_module, z, T,  compute_smallest=False, largest=None):
    n = len(z)
    for i in range(T):
        za = linear_module.multiply(*z)
        zb = linear_module.multiply_transpose(*z)
        if compute_smallest:
            zn = tuple(-2*largest*a + 0.5*b + 0.5*c for a,b,c in zip(z, za, zb))
        else:
            zn = tuple(0.5*a + 0.5*b for a,b in zip(za, zb))
        x = sum((zn[i]*z[i]).sum().item() for i in range(n))
        y = sum((z[i]*z[i]).sum().item() for i in range(n))
        lam = x/y
        z = tuple(zn[i]/np.sqrt(y) for i in range(n))
    return lam + 2 * largest if compute_smallest else lam


def get_splitting_stats(dataLoader, model):
    model = cuda(model)
    model.train()
    model.mon.save_abs_err = True
    for batch in dataLoader:
        data, target = cuda(batch[0]), cuda(batch[1])
        model(data)
        return model.mon.errs


def HCAS_loader(pra: int,tau: int, train_batch_size:int, test_batch_size: Optional[int]=None, normalize: Optional[bool]=True, test_fraction:Optional[float]=0.1):
        if test_batch_size is None:
            test_batch_size = train_batch_size

        ver = 6  # Neural network version
        table_ver = 6  # Table Version
        trainingDataFiles = "./TrainingData/HCAS_rect_TrainingData_v%d_pra%d_tau%02d.h5"  # File format for training data

        print("Loading Data for HCAS, pra %02d, Network Version %d" % (pra, ver))
        f = h5py.File(trainingDataFiles % (table_ver, pra, tau), 'r')
        X_train = np.array(f['X'])
        X_train = X_train - X_train.min(0)
        assert (X_train.max(0)==1).all()
        Q = np.array(f['y'])
        means = np.array(f['means'])
        ranges = np.array(f['ranges'])
        mins = np.array(f['min_inputs'])
        maxes = np.array(f['max_inputs'])
        widths = maxes-mins

        mean = X_train.mean(0)
        std = X_train.std(0)

        if normalize:
            X_train = (X_train - mean)/std

        X_train = torch.tensor(X_train, dtype=torch.float32)
        Q = torch.tensor(Q)

        # a=Q.argmax(0)
        # for label in range(5):
        #     if label == 0:  # COC clear of contact
        #         c = "black"
        #     elif label == 1:  # Weak Left
        #         c = "blue"
        #     elif label == 3:  # Strong Left
        #         c = "purple"
        #     elif label == 2:  # Weak Right
        #         c = "green"
        #     elif label == 4:  # Strong Right
        #         c = "gray"
        #     plt.scatter(X_train[((a == label).__and__((X_train[:, 2] - 0.25).abs() < 0.01))][:, 0],
        #                 X_train[((a == label).__and__((X_train[:, 2] - 0.25).abs() < 0.01))][:, 1], color=c)
        # plt.show()

        n_samples = X_train.shape[0]
        n_test = int(np.ceil(n_samples*test_fraction))
        np.random.seed(42)
        test_id = np.random.choice(n_samples, n_test, replace=False)
        train_mask = torch.ones(n_samples, dtype=torch.bool)
        train_mask[test_id] = False
        test_mask = ~train_mask


        train_ds = TensorDataset(X_train[train_mask], Q[train_mask])
        test_ds = TensorDataset(X_train[test_mask], Q[test_mask])

        trainLoader = DataLoader(train_ds,
                        batch_size=train_batch_size,
                        shuffle=True)

        if test_fraction>0:
            testLoader = DataLoader(test_ds,
                        batch_size=test_batch_size,
                        shuffle=True)
        else:
            testLoader = trainLoader

        return trainLoader, testLoader, mean, std


def asymMSE(y_true, y_pred):
    lossFactor = 40.0
    numOut = y_true.shape[1]
    d = y_true-y_pred
    maxes = y_true.argmax(1)
    maxes_onehot = torch.nn.functional.one_hot(maxes,numOut)
    others_onehot = maxes_onehot-1
    d_opt = d*maxes_onehot
    d_sub = d*others_onehot
    b = d_opt.square()
    d = d_sub.square()
    a = lossFactor*(numOut-1)*(b+d_opt.abs())
    c = lossFactor*(d+d_sub.abs())

    loss = torch.where(d_sub>0, c, d) + torch.where(d_opt>0,a,b)
    return loss