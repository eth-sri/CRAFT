import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from utils import cuda
from monDEQ.utils import asymMSE
import time

from attacks import adv_whitebox

"""
Based on: https://github.com/locuslab/monotone_op_net/blob/master/train.py
"""

def eval_model(testLoader, model, tune_alpha=False, max_alpha=0.2):
    nat_acc = 0
    n_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(testLoader):
            data, target = cuda(batch[0]), cuda(batch[1])

            if batch_idx == 0 and tune_alpha:
                run_tune_alpha(model, data, max_alpha)

            preds = model(data)
            if target.dim() == 1 or target.shape[1]==1:
                correct = preds.float().argmax(1).eq(target.data).sum()
            else:
                correct = preds.float().argmax(1).eq(target.data.argmax(1)).sum()
            nat_acc += correct
            n_samples += len(target)
    print(f"Accuracy: {nat_acc/n_samples:.4f}")


def train(trainLoader, testLoader, model, epochs=15, max_lr=1e-3,
          print_freq=10, change_mo=True, model_path=None, lr_mode='step',
          step=10, tune_alpha=False, max_alpha=1., eps=None, args=None, std=1, mean=0):

    optimizer = optim.Adam(model.parameters(), lr=max_lr)

    if lr_mode == '1cycle':
        lr_schedule = lambda t: np.interp([t],
                                          [0, (epochs-5)//2, epochs-5, epochs],
                                          [1e-3, max_lr, 1e-3, 1e-3])[0]
    elif lr_mode == 'step':
        lr_scheduler =optim.lr_scheduler.StepLR(optimizer, step, gamma=0.1, last_epoch=-1)
    elif lr_mode != 'constant':
        raise Exception('lr mode one of constant, step, 1cycle')

    if change_mo:
        max_mo = 0.85
        momentum_schedule = lambda t: np.interp([t],
                                                [0, (epochs - 5) // 2, epochs - 5, epochs],
                                                [0.95, max_mo, 0.95, 0.95])[0]

    model = model.to("cuda")

    for epoch in range(1, 1 + epochs):
        nProcessed = 0
        nTrain = len(trainLoader.dataset)
        model.train()
        start = time.time()
        for batch_idx, batch in enumerate(trainLoader):
            if (batch_idx  == 30 or batch_idx == int(len(trainLoader)/2)) and tune_alpha:
                run_tune_alpha(model, cuda(batch[0]), max_alpha)
            if lr_mode == '1cycle':
                lr = lr_schedule(epoch -  1 + batch_idx/ len(trainLoader))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            if change_mo:
                beta1 = momentum_schedule(epoch - 1 + batch_idx / len(trainLoader))
                for param_group in optimizer.param_groups:
                    param_group['betas'] = (beta1, optimizer.param_groups[0]['betas'][1])

            data, target = cuda(batch[0]), cuda(batch[1])
            optimizer.zero_grad()
            loss = 0.

            if eps is not None:
                if not isinstance(mean, torch.Tensor):
                    mean = torch.tensor(mean).to(data.device)
                    std = torch.tensor(std).to(data.device)
                specLB = torch.maximum(data - eps/std, -mean/std)
                specUB = torch.minimum(data + eps/std, (1-mean)/std)
                model.eval()
                adex = adv_whitebox(model, data, target, specLB, specUB, data.device, num_steps=8, step_size=0.25,
                             ODI_num_steps=0, ODI_step_size=1., lossFunc="margin", restarts=1, train=True)
                model.train()
                optimizer.zero_grad()
                preds_adv = model(adex)
                loss += nn.CrossEntropyLoss()(preds_adv, target)
                incorrect_adv = preds_adv.float().argmax(1).ne(target.data).sum()
                err_adv = 100. * incorrect_adv.float() / float(len(data))
            else:
                err_adv = -100.

            preds = model(data)
            if target.dim() == 1 or target.shape[1] == 1:
                loss += nn.CrossEntropyLoss()(preds, target)
            else:
                loss += asymMSE(target, preds).mean()


            loss.backward()
            nProcessed += len(data)
            if batch_idx % print_freq == 0 and batch_idx > 0:
                if target.dim() == 1 or target.shape[1] == 1:
                    incorrect = preds.float().argmax(1).ne(target.data).sum()
                else:
                    incorrect = preds.float().argmax(1).ne(target.data.argmax(1)).sum()
                err = 100. * incorrect.float() / float(len(data))
                partialEpoch = epoch + batch_idx / len(trainLoader) - 1
                print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tError: {:.2f} \tError Adv: {:.2f} \tLR: {:.2e}'.format(
                    partialEpoch, nProcessed, nTrain,
                    100. * batch_idx / len(trainLoader),
                    loss.item(), err, err_adv, lr_scheduler.get_last_lr()[0]))
                model.mon.stats.report()
                model.mon.stats.reset()
            optimizer.step()

        if lr_mode == 'step':
            lr_scheduler.step()

        if model_path is not None:
            torch.save(model.state_dict(), model_path)

        print("Tot train time: {}".format(time.time() - start))

        start = time.time()
        test_loss = 0
        incorrect = 0
        test_loss_adv = 0
        incorrect_adv = 0
        model.eval()
        with torch.no_grad():
            for batch in testLoader:
                model.eval()
                data, target = cuda(batch[0]), cuda(batch[1])
                preds = model(data)
                if target.dim() == 1 or target.shape[1] == 1:
                    loss = nn.CrossEntropyLoss(reduction='sum')(preds, target)
                else:
                    loss = asymMSE(target, preds).mean()
                test_loss += loss
                if target.dim() == 1 or target.shape[1] == 1:
                    incorrect += preds.float().argmax(1).ne(target.data).sum()
                else:
                    incorrect += preds.float().argmax(1).ne(target.data.argmax(1)).sum()

                if eps is not None:
                    specLB = torch.maximum(data - eps / std, -mean / std)
                    specUB = torch.minimum(data + eps / std, (1 - mean) / std)
                    adex = adv_whitebox(model, data, target, specLB, specUB, data.device, num_steps=8,
                                        step_size=0.25,
                                        ODI_num_steps=0, ODI_step_size=1., lossFunc="margin", restarts=1,
                                        train=True)
                    optimizer.zero_grad()
                    preds_adv = model(adex)
                    test_loss_adv += nn.CrossEntropyLoss()(preds_adv, target)
                    incorrect_adv += preds_adv.float().argmax(1).ne(target.data).sum()
            model.train()
            test_loss /= len(testLoader.dataset)
            test_loss_adv /= len(testLoader.dataset)

            nTotal = len(testLoader.dataset)
            err = 100. * incorrect.float() / float(nTotal)
            err_adv = 100. * incorrect_adv.float() / float(nTotal)

            print('\n\nTest set: Average loss: {:.4f}, Average loss adv: {:.4f}, Error: {}/{} ({:.2f}%),  Error_adv: {}/{} ({:.2f}%)'.format(
                test_loss, test_loss_adv, incorrect, nTotal, err, incorrect_adv, nTotal, err_adv))

        print("Tot test time: {}\n\n\n\n".format(time.time() - start))
        adv_string = "" if eps is None else f"_adv_e{eps*100:2}_"
        if args is not None:
            if args.conv:
                path = f"models/{args.prefix}_mon_conv_c{args.hidden}_k{args.kernel_x}_s{args.stride_x}_m{args.monotone}{adv_string}.pt"
            else:
                path = f"models/{args.prefix}_mon_h{args.hidden}_m{args.monotone}{adv_string}.pt"
            torch.save(model.state_dict(), path)


def run_tune_alpha(model, x, max_alpha):
    print("----tuning alpha----")
    print("current: ", model.mon.alpha)
    orig_alpha  =  model.mon.alpha
    model.mon.stats.reset()
    model.mon.alpha = max_alpha
    with torch.no_grad():
        model(x)
    iters = model.mon.stats.fwd_iters.val
    model.mon.stats.reset()
    iters_n = iters
    print('alpha: {}\t iters: {}'.format(model.mon.alpha, iters_n))
    while model.mon.alpha > 1e-4 and iters_n <= iters:
        model.mon.alpha = model.mon.alpha*0.8
        with torch.no_grad():
            model(x)
        iters = iters_n
        iters_n = model.mon.stats.fwd_iters.val
        print('alpha: {}\t iters: {}'.format(model.mon.alpha, iters_n))
        model.mon.stats.reset()

    if iters==model.mon.max_iter:
        print("none converged, resetting to current")
        model.mon.alpha=orig_alpha
    else:
        model.mon.alpha = model.mon.alpha / 0.8
        print("setting to: ", model.mon.alpha)
    print("--------------\n")


def mnist_loaders(train_batch_size, test_batch_size=None, normalize=True, augment=False):
    if test_batch_size is None:
        test_batch_size = train_batch_size

    if normalize:
        mean = [0.1307,]
        std = [0.3081,]
        normalize = [transforms.Normalize(mean=mean, std=std)]
    else:
        mean = None
        std = None
        normalize = []

    if augment:
        transforms_list_train = [transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(28, 4),
                            transforms.ToTensor(),] + normalize
    else:
        transforms_list_train = [transforms.ToTensor(),] + normalize

    transform_train = transforms.Compose(transforms_list_train)
    transform_test = transforms.Compose([transforms.ToTensor(),] + normalize)

    trainLoader = torch.utils.data.DataLoader(
        dset.MNIST('data',
                train=True,
                download=True,
                transform=transform_train),
        batch_size=train_batch_size,
        shuffle=True)

    testLoader = torch.utils.data.DataLoader(
        dset.MNIST('data',
                train=False,
                transform=transform_test),
        batch_size=test_batch_size,
        shuffle=False)

    return trainLoader, testLoader, mean, std


def cifar_loaders(train_batch_size, test_batch_size=None, normalize=True, augment=True):
    if test_batch_size is None:
        test_batch_size = train_batch_size

    if normalize:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        normalize = [transforms.Normalize(mean=mean, std=std)]
    else:
        mean = None
        std = None
        normalize = []

    if augment:
        transforms_list_train = [transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32, 4),
                            transforms.ToTensor(),] + normalize
    else:
        transforms_list_train = [transforms.ToTensor(),] + normalize

    transform_train = transforms.Compose(transforms_list_train)
    transform_test = transforms.Compose([transforms.ToTensor(),] + normalize)

    train_dset = dset.CIFAR10('data',
                              train=True,
                              download=True,
                              transform=transform_train)
    test_dset = dset.CIFAR10('data',
                             train=False,
                             transform=transform_test)

    trainLoader = torch.utils.data.DataLoader(train_dset, batch_size=train_batch_size,
                                              shuffle=True, pin_memory=True)

    testLoader = torch.utils.data.DataLoader(test_dset, batch_size=test_batch_size,
                                             shuffle=False, pin_memory=True)

    return trainLoader, testLoader, mean, std


def svhn_loaders(train_batch_size, test_batch_size=None):
    if test_batch_size is None:
        test_batch_size = train_batch_size

    normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                      std=[0.1980, 0.2010, 0.1970])
    train_loader = torch.utils.data.DataLoader(
            dset.SVHN(
                root='data', split='train', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize
                ]),
            ),
            batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dset.SVHN(
            root='data', split='test', download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])),
        batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader
