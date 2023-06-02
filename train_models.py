import re

import torch
import numpy as np
import scipy.io as sio

from monDEQ.train import mnist_loaders, train, cifar_loaders, eval_model
import monDEQ.splitting as sp
from mondeq_nets import SingleFcNet, SingleConvNet
from monDEQ.utils import HCAS_loader
import argparse

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch-size", type=int, required=True, default=128, help="Batch size")
    parser.add_argument("-vs", "--val-size", type=int, required=True, default=400, help="Validation size")
    parser.add_argument("-hi", "--hidden", type=int, required=True, default=200, help="hidden dimension")
    parser.add_argument("-m", "--monotone", type=float, required=True, default=20, help="Monotonicity parameter")
    parser.add_argument("-pr", "--prefix", type=str, default="", help="Export prefix")
    parser.add_argument("-ds", "--dataset", type=str, default="mnist", help="Dataset to use")
    parser.add_argument("-conv", "--conv", action="store_true", default=False, help="Use a conv net")
    parser.add_argument("-k", "--kernel_x", type=int, default=6, help="Kernel size for input conv")
    parser.add_argument("-s", "--stride_x", type=int, default=3, help="Stride for input conv")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-p", "--path", type=str, default=None, required=False, help="Path to the model w.r.t. directory root.")
    parser.add_argument("-eval", "--eval", action="store_true", default=False, help="Just eval the model")
    parser.add_argument("-eps", "--eps",  type=float, default=None, help="epsilon for adversarial training")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size
    val_size = args.val_size
    hidden_dim = args.hidden
    m = args.monotone
    dataset = args.dataset
    k = args.kernel_x
    s = args.stride_x

    if dataset == "mnist":
        # mean = [0.1307,]
        # std = [0.3081,]
        shp_d = [1, 28, 28]
        trainLoader, testLoader, mean, std = mnist_loaders(train_batch_size=batch_size, test_batch_size=val_size, normalize=True, augment=False)
        n_class = 10
    elif dataset == "cifar":
        # mean = [0.4914, 0.4822, 0.4465]
        # std = [0.2470, 0.2435, 0.2616]
        shp_d = [3, 32, 32]
        trainLoader, testLoader,  mean, std = cifar_loaders(train_batch_size=batch_size, test_batch_size=val_size, normalize=True, augment=True)
        n_class = 10
    elif dataset.startswith("HCAS"):
        # mean = [0.4914, 0.4822, 0.4465]
        # std = [0.2470, 0.2435, 0.2616]
        match = re.match("HCAS_p([0-9])_t([0-9]*)",dataset)
        if match is None:
            pra = 0
            tau = 10
        else:
            pra = int(match.group(1))
            tau = int(match.group(2))
        shp_d = [3]
        trainLoader, testLoader,  mean, std = HCAS_loader(pra=pra, tau=tau, train_batch_size=batch_size, test_batch_size=val_size, normalize=True, test_fraction=0)
        n_class = 5
    else:
        assert False, f"Dataset {dataset} is unknown"

    if args.conv:
        model = SingleConvNet(sp.MONPeacemanRachford, in_dim=shp_d[-1], alpha=1.0,
                              in_channels=shp_d[0],
                              out_channels=hidden_dim,
                              m=m, kernel_x=k, stride_x=s,
                              tol=1e-6, max_iter=300, n_class=n_class)
    else:
        if args.path is not None and args.path.startswith("mat_models"):
            model = SingleFcNet(sp.MONPeacemanRachford, in_dim=np.prod(shp_d), latent_dim=87, alpha=0.2, max_iter=300,
                                tol=1e-7, m=m, n_class=n_class)
            # model.load_state_dict(torch.load("models/mon_h40_m20.0.pt", map_location=torch.device('cpu')))
            mat_contents = sio.loadmat(args.path)
            # Mapping
            # U = U
            model.mon.linear_module.U.weight = torch.nn.Parameter(torch.Tensor(mat_contents['U']))
            model.mon.linear_module.U.bias = torch.nn.Parameter(torch.Tensor(mat_contents['u']).flatten())
            # A = A
            model.mon.linear_module.A.weight = torch.nn.Parameter(torch.Tensor(mat_contents['A']))
            # B = B
            model.mon.linear_module.B.weight = torch.nn.Parameter(torch.Tensor(mat_contents['B']))
            # W = C
            model.Wout.weight = torch.nn.Parameter(torch.Tensor(mat_contents['C']))
            model.Wout.bias = torch.nn.Parameter(torch.Tensor(mat_contents['c']).flatten())

            print("Loaded Matlab model")
        else:
            model = SingleFcNet(sp.MONPeacemanRachford, in_dim=np.prod(shp_d), latent_dim=hidden_dim, alpha=1.0,
                                max_iter=300, tol=1e-6, m=m, n_class=n_class)  #parameter which controls the strong monotonicity of W

    if args.path is not None and not args.path.startswith("mat_models"):
        unexpected, extra = model.load_state_dict(torch.load(args.path, map_location=torch.device(device)), strict=False)
        assert len(extra) == 0
        if args.eval:
            model.eval()

    model.to(device)

    if args.eval:
        eval_model(testLoader, model, tune_alpha=True, max_alpha=0.2)
    else:
        train(trainLoader, testLoader,
            model,
            max_lr=1e-3,
            lr_mode='step',  #use step decay learning rate
            step=int(np.floor(args.epochs*0.4)),
            change_mo=False, #do not adjust momentum during training
            epochs=args.epochs,
            print_freq=100,
            tune_alpha=True, args=args,
            max_alpha=0.5,
            eps=args.eps,
            mean=mean,
            std=std)


    # Save model
    # path = f"models/{args.prefix}_mon_h{hidden_dim}_m{m}.pt"
    # torch.save(model.state_dict(), path)
