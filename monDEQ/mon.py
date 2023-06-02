import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ai.concrete_layers import Bias
from ai.abstract_layers import Sequential

"""
Based on: https://github.com/locuslab/monotone_op_net/blob/master/mon.py
"""


class MONSingleFc(nn.Module):
    """ Simple MON linear class, just a single full multiply. """

    def __init__(self, in_dim, out_dim, m=1.0):
        super().__init__()
        self.U = nn.Linear(in_dim, out_dim)
        self.A = nn.Linear(out_dim, out_dim, bias=False)
        self.B = nn.Linear(out_dim, out_dim, bias=False)
        self.m = m
        self.out_dim = [out_dim]

    def x_shape(self, n_batch):
        return (n_batch, self.U.in_features)

    def z_shape(self, n_batch):
        return ((n_batch, self.A.in_features),)

    def forward(self, x, *z):
        return (self.U(x) + self.multiply(*z)[0],)

    def bias(self, x):
        return (self.U(x),)

    def multiply(self, *z):
        ATAz = self.A(z[0]) @ self.A.weight
        z_out = (1 - self.m) * z[0] - ATAz + self.B(z[0]) - z[0] @ self.B.weight
        return (z_out,)

    def multiply_transpose(self, *g):
        ATAg = self.A(g[0]) @ self.A.weight
        g_out = (1 - self.m) * g[0] - ATAg - self.B(g[0]) + g[0] @ self.B.weight
        return (g_out,)

    def get_W(self):
        I = torch.eye(self.A.weight.shape[0], dtype=self.A.weight.dtype,
                      device=self.A.weight.device)
        W = (1 - self.m) * I - self.A.weight.T @ self.A.weight + self.B.weight - self.B.weight.T
        return W

    def init_inverse(self, alpha, beta):
        I = torch.eye(self.A.weight.shape[0], dtype=self.A.weight.dtype,
                      device=self.A.weight.device)
        W = (1 - self.m) * I - self.A.weight.T @ self.A.weight + self.B.weight - self.B.weight.T
        self.Winv = torch.inverse(alpha * I + beta * W)#.detach()

    def inverse(self, *z):
        return (z[0] @ self.Winv.transpose(0, 1),)

    def inverse_transpose(self, *g):
        return (g[0] @ self.Winv,)

    def _forward_abs(self, x, z):
        if not hasattr(self, 'W'): self.W = None
        if not hasattr(self, 'Ux'): self.Ux = None; self.Ux_id = id(x)
        if self.W is None:
            I = torch.eye(self.A.weight.shape[0], dtype=self.A.weight.dtype,
                          device=self.A.weight.device)
            self.W = ((1 - self.m) * I - self.A.weight.T @ self.A.weight + self.B.weight - self.B.weight.T).detach()
        if self.Ux is None or self.Ux_id != id(x):
            self.Ux = (x.linear(self.U.weight, self.U.bias)).detach()
            self.Ux_id = id(x)

        Wz = z.linear(self.W, bias=None)
        return Wz + self.Ux

    # def _forward_abs_2(self, x1, x2, z1, z2):
    #     if not hasattr(self, 'W'): self.W = None
    #     if not hasattr(self, 'Ux1'): self.Ux1 = None; self.Ux1_id = id(x1)
    #     if not hasattr(self, 'Ux2'): self.Ux2 = None; self.Ux2_id = id(x2)
    #     if self.W is None:
    #         I = torch.eye(self.A.weight.shape[0], dtype=self.A.weight.dtype,
    #                       device=self.A.weight.device)
    #         self.W = (1 - self.m) * I - self.A.weight.T @ self.A.weight + self.B.weight - self.B.weight.T
    #     if self.Ux1 is None or self.Ux1_id != id(x1):
    #         self.Ux1 = x1.linear(self.U.weight, self.U.bias)
    #         self.Ux1_id = id(x1)
    #     if self.Ux2 is None or self.Ux2_id != id(x2):
    #         self.Ux2 = x2.linear(self.U.weight, self.U.bias)
    #         self.Ux2_id = id(x2)
    #
    #     Wz1 = z1.linear(self.W)
    #     Wz2 = z2.linear(self.W)
    #
    #     t1 = Wz1 + self.Ux1
    #     t2 = Wz2 + self.Ux2
    #
    #     return Wz1 + self.Ux1

    def reset(self):
        if hasattr(self, 'Ux1'): self.Ux1 = None
        if hasattr(self, 'Ux2'): self.Ux2 = None
        if hasattr(self, 'Ux'): self.Ux = None
        if hasattr(self, 'Ux1_id'): self.Ux1_id = None
        if hasattr(self, 'Ux2_id'): self.Ux2_id = None
        if hasattr(self, 'Ux_id'): self.Ux_id = None



class MONReLU(nn.Module):
    def forward(self, *z):
        return tuple(F.relu(z_) for z_ in z)

    def derivative(self, *z):
        return tuple((z_ > 0).type_as(z[0]) for z_ in z)

    def _forward_abs(self, x, deepz_lambda=None):
        return x.relu(deepz_lambda, None, None)[0]


# Convolutional layers w/ FFT-based inverses

def fft_to_complex_matrix(x):
    """ Create matrix with [a -b; b a] entries for complex numbers. """
    x_stacked = torch.stack((x, torch.flip(x, (4,))), dim=5).permute(2, 3, 0, 4, 1, 5)
    x_stacked[:, :, :, 0, :, 1] *= -1
    return x_stacked.reshape(-1, 2 * x.shape[0], 2 * x.shape[1])


def fft_to_complex_vector(x):
    """ Create stacked vector with [a;b] entries for complex numbers"""
    return x.permute(2, 3, 0, 1, 4).reshape(-1, x.shape[0], x.shape[1] * 2)


def init_fft_conv(weight, hw):
    """ Initialize fft-based convolution.

    Args:
        weight: Pytorch kernel
        hw: (height, width) tuple
    """
    px, py = (weight.shape[2] - 1) // 2, (weight.shape[3] - 1) // 2
    kernel = torch.flip(weight, (2, 3))
    kernel = F.pad(F.pad(kernel, (0, hw[0] - weight.shape[2], 0, hw[1] - weight.shape[3])),
                   (0, py, 0, px), mode="circular")[:, :, py:, px:]
    return fft_to_complex_matrix(torch.rfft(kernel, 2, onesided=False))


def fft_conv(x, w_fft, transpose=False):
    """ Perhaps FFT-based circular convolution.

    Args:
        x: (B, C, H, W) tensor
        w_fft: conv kernel processed by init_fft_conv
        transpose: flag of whether to transpose convolution
    """
    x_fft = fft_to_complex_vector(torch.rfft(x, 2, onesided=False))
    wx_fft = x_fft.bmm(w_fft.transpose(1, 2)) if not transpose else x_fft.bmm(w_fft)
    wx_fft = wx_fft.view(x.shape[2], x.shape[3], wx_fft.shape[1], -1, 2).permute(2, 3, 0, 1, 4)
    return torch.irfft(wx_fft, 2, onesided=False)


class MONSingleConv(nn.Module):
    """ MON class with a single 3x3 (circular) convolution """

    def __init__(self, in_channels, out_channels, shp, kernel_size=3, m=1.0, stride=1, kernel_x=None):
        super().__init__()
        kernel_x = kernel_size if kernel_x is None else kernel_x
        self.U = nn.Conv2d(in_channels, out_channels, kernel_x, stride=stride)
        out_dim = getShapeConv((in_channels, shp[-2], shp[-1]), (in_channels, kernel_x, kernel_x),
                     stride=stride, padding=1)

        self.A = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
        self.g = nn.Parameter(torch.tensor(1.))
        self.h = nn.Parameter(torch.tensor(1.))
        self.B = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
        self.pad = 4 * ((kernel_size - 1) // 2,)
        self.shp = shp
        self.z_shp = out_dim[1:]
        self.m = m
        self.out_dim = list(out_dim)


    def cpad(self, x):
        if isinstance(x, torch.Tensor):
            return F.pad(x, self.pad, mode="circular")
        else:
            return x.pad( self.pad, mode="circular")

    def uncpad(self, x):
        return x[:, :, 2 * self.pad[0]:-2 * self.pad[1], 2 * self.pad[2]:-2 * self.pad[3]]

    def x_shape(self, n_batch):
        return (n_batch, self.U.in_channels, self.shp[0], self.shp[1])

    def z_shape(self, n_batch):
        return ((n_batch, self.A.in_channels, self.z_shp[0], self.z_shp[1]),)

    def forward(self, x, *z):
        # circular padding is broken in PyTorch
        return (F.conv2d(self.cpad(x), self.U.weight, self.U.bias, self.U.stride, self.U.padding, self.U.dilation, self.U.groups) + self.multiply(*z)[0],)

    def _forward_abs(self, x, z):
        if not hasattr(self, 'W'): self.W = None
        if not hasattr(self, 'Ux'): self.Ux = None; self.Ux_id = id(x)
        if self.Ux is None or self.Ux_id != id(x):
            self.Ux =self.cpad(x).conv2d(self.U.weight, self.U.bias, self.U.stride, self.U.padding, self.U.dilation, self.U.groups)
            self.Ux_id = id(x)

        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.h * self.B.weight / self.B.weight.view(-1).norm()
        Az = self.cpad(z).conv2d(A, None, self.A.stride, self.A.padding, self.A.dilation, self.A.groups)
        ATAz = self.uncpad(self.cpad(Az).conv_transpose2d(A, self.A.bias, self.A.stride, self.A.padding, self.A.output_padding, self.A.groups, self.A.dilation))
        Bz = self.cpad(z).conv2d(B, self.B.bias, self.B.stride, self.B.padding, self.B.dilation, self.B.groups)
        BTz = self.uncpad(self.cpad(z).conv_transpose2d(B, self.B.bias, self.B.stride, self.B.padding, self.B.output_padding, self.B.groups, self.B.dilation))
        z_out = (1 - self.m) * z - self.g * ATAz + Bz - BTz + self.Ux

        return z_out

    def bias(self, x):
        return (F.conv2d(self.cpad(x), self.U.weight, self.U.bias, self.U.stride, self.U.padding, self.U.dilation, self.U.groups),)

    def multiply(self, *z):
        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.h * self.B.weight / self.B.weight.view(-1).norm()
        Az = F.conv2d(self.cpad(z[0]), A)
        ATAz = self.uncpad(F.conv_transpose2d(self.cpad(Az), A))
        Bz = F.conv2d(self.cpad(z[0]), B)
        BTz = self.uncpad(F.conv_transpose2d(self.cpad(z[0]), B))
        z_out = (1 - self.m) * z[0] - self.g * ATAz + Bz - BTz
        return (z_out,)

    def multiply_transpose(self, *g):
        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.h * self.B.weight / self.B.weight.view(-1).norm()
        Ag = F.conv2d(self.cpad(g[0]), A)
        ATAg = self.uncpad(F.conv_transpose2d(self.cpad(Ag), A))
        Bg = F.conv2d(self.cpad(g[0]), B)
        BTg = self.uncpad(F.conv_transpose2d(self.cpad(g[0]), B))
        g_out = (1 - self.m) * g[0] - self.g * ATAg - Bg + BTg
        return (g_out,)

    def init_inverse(self, alpha, beta):
        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.h * self.B.weight / self.B.weight.view(-1).norm()
        Afft = init_fft_conv(A, self.z_shp)
        Bfft = init_fft_conv(B, self.z_shp)
        I = torch.eye(Afft.shape[1], dtype=Afft.dtype,
                      device=Afft.device)[None, :, :]
        self.Wfft = (1 - self.m) * I - self.g * Afft.transpose(1, 2) @ Afft + Bfft - Bfft.transpose(1, 2)
        self.Winv = torch.inverse(alpha * I + beta * self.Wfft)

    def inverse(self, *z):
        return (fft_conv(z[0], self.Winv),)

    def inverse_transpose(self, *g):
        return (fft_conv(g[0], self.Winv, transpose=True),)

    def reset(self):
        if hasattr(self, 'Ux'): self.Ux = None


class MONBorderReLU(nn.Module):
    def __init__(self, border=1):
        super().__init__()
        self.border = border

    def forward(self, *z):
        zn = tuple(F.relu(z_) for z_ in z)
        for i in range(len(zn)):
            zn[i][:, :, :self.border, :] = 0
            zn[i][:, :, -self.border:, :] = 0
            zn[i][:, :, :, :self.border] = 0
            zn[i][:, :, :, -self.border:] = 0
        return zn

    def derivative(self, *z):
        return tuple((z_ > 0).type_as(z[0]) for z_ in z)

    def _forward_abs(self, z, deepz_lambda=None):
        zn = z.border_relu(deepz_lambda, None, None, self.border)
        return zn

class MONMultiConv(nn.Module):
    def __init__(self, in_channels, conv_channels, image_size, kernel_size=3, m=1.0):
        super().__init__()
        self.pad = 4 * ((kernel_size - 1) // 2,)
        self.conv_shp = tuple((image_size - 2 * self.pad[0]) // 2 ** i + 2 * self.pad[0]
                              for i in range(len(conv_channels)))
        self.m = m

        # create convolutional layers
        self.U = nn.Conv2d(in_channels, conv_channels[0], kernel_size)
        self.A0 = nn.ModuleList([nn.Conv2d(c, c, kernel_size, bias=False) for c in conv_channels])
        self.B0 = nn.ModuleList([nn.Conv2d(c, c, kernel_size, bias=False) for c in conv_channels])
        self.A_n0 = nn.ModuleList([nn.Conv2d(c1, c2, kernel_size, bias=False, stride=2)
                                   for c1, c2 in zip(conv_channels[:-1], conv_channels[1:])])

        self.g = nn.ParameterList([nn.Parameter(torch.tensor(1.)) for _ in range(len(conv_channels))])
        self.gn = nn.ParameterList([nn.Parameter(torch.tensor(1.)) for _ in range(len(conv_channels) - 1)])
        self.h = nn.ParameterList([nn.Parameter(torch.tensor(1.)) for _ in range(len(conv_channels))])



        self.S_idx = list()
        self.S_idxT = list()
        for n in self.conv_shp:
            p = n // 2
            q = n
            idxT = list()
            _idx = [[j + (i - 1) * p for i in range(1, q + 1)] for j in range(1, p + 1)]
            for i in _idx:
                for j in i:
                    idxT.append(j - 1)
            _idx = [[j + (i - 1) * p + p * q for i in range(1, q + 1)] for j in range(1, p + 1)]
            for i in _idx:
                for j in i:
                    idxT.append(j - 1)
            idx = list()
            _idx = [[j + (i - 1) * q for i in range(1, p + 1)] for j in range(1, q + 1)]
            for i in _idx:
                for j in i:
                    idx.append(j - 1)
            _idx = [[j + (i - 1) * q + p * q for i in range(1, p + 1)] for j in range(1, q + 1)]
            for i in _idx:
                for j in i:
                    idx.append(j - 1)
            self.S_idx.append(idx)
            self.S_idxT.append(idxT)

    def A(self, i):
        return torch.sqrt(self.g[i]) * self.A0[i].weight / self.A0[i].weight.view(-1).norm()

    def A_n(self, i):
        return torch.sqrt(self.gn[i]) * self.A_n0[i].weight / self.A_n0[i].weight.view(-1).norm()

    def B(self, i):
        return self.h[i] * self.B0[i].weight / self.B0[i].weight.view(-1).norm()

    def cpad(self, x):
        return F.pad(x, self.pad, mode="circular")

    def uncpad(self, x):
        return x[:, :, 2 * self.pad[0]:-2 * self.pad[1], 2 * self.pad[2]:-2 * self.pad[3]]

    def zpad(self, x):
        return F.pad(x, (0, 1, 0, 1))

    def unzpad(self, x):
        return x[:, :, :-1, :-1]

    def unstride(self, x):
        x[:, :, :, -1] += x[:, :, :, 0]
        x[:, :, -1, :] += x[:, :, 0, :]
        return x[:, :, 1:, 1:]

    def x_shape(self, n_batch):
        return (n_batch, self.U.in_channels, self.conv_shp[0], self.conv_shp[0])

    def z_shape(self, n_batch):
        return tuple((n_batch, self.A0[i].in_channels, self.conv_shp[i], self.conv_shp[i])
                     for i in range(len(self.A0)))

    def forward(self, x, *z):
        z_out = self.multiply(*z)
        bias = self.bias(x)
        return tuple([z_out[i] + bias[i] for i in range(len(self.A0))])

    def bias(self, x):
        z_shape = self.z_shape(x.shape[0])
        n = len(self.A0)

        b_out = [self.U(self.cpad(x))]
        for i in range(n - 1):
            b_out.append(torch.zeros(z_shape[i + 1], dtype=self.A0[0].weight.dtype,
                   device=self.A0[0].weight.device))
        return tuple(b_out)

    def multiply(self, *z):

        def multiply_zi(z1, A1, B1, A1_n=None, z0=None, A2_n=None):
            Az1 = F.conv2d(self.cpad(z1), A1)
            A1TA1z1 = self.uncpad(F.conv_transpose2d(self.cpad(Az1), A1))
            B1z1 = F.conv2d(self.cpad(z1), B1)
            B1Tz1 = self.uncpad(F.conv_transpose2d(self.cpad(z1), B1))
            out = (1 - self.m) * z1 - A1TA1z1 + B1z1 - B1Tz1
            if A2_n is not None:
                A2_nz1 = F.conv2d(self.cpad(z1), A2_n, stride=2)
                A2_nTA2_nz1 = self.unstride(F.conv_transpose2d(A2_nz1,
                                                               A2_n, stride=2))
                out -= A2_nTA2_nz1
            if A1_n is not None:
                A1_nz0 = self.zpad(F.conv2d(self.cpad(z0), A1_n, stride=2))
                A1TA1_nz0 = self.uncpad(F.conv_transpose2d(self.cpad(A1_nz0), A1))
                out -= 2 * A1TA1_nz0
            return out

        n = len(self.A0)
        z_out = [multiply_zi(z[0], self.A(0), self.B(0), A2_n=self.A_n(0))]
        for i in range(1, n - 1):
            z_out.append(multiply_zi(z[i], self.A(i), self.B(i),
                                     A1_n=self.A_n(i - 1), z0=z[i - 1], A2_n=self.A_n(i)))
        z_out.append(multiply_zi(z[n - 1], self.A(n - 1), self.B(n - 1),
                                 A1_n=self.A_n(n - 2), z0=z[n - 2]))

        return tuple(z_out)

    def multiply_transpose(self, *g):

        def multiply_zi(z1, A1, B1, z2=None, A2_n=None, A2=None):
            Az1 = F.conv2d(self.cpad(z1), A1)
            A1TA1z1 = self.uncpad(F.conv_transpose2d(self.cpad(Az1), A1))
            B1z1 = F.conv2d(self.cpad(z1), B1)
            B1Tz1 = self.uncpad(F.conv_transpose2d(self.cpad(z1), B1))
            out = (1 - self.m) * z1 - A1TA1z1 - B1z1 + B1Tz1
            if A2_n is not None:
                A2z2 = F.conv2d(self.cpad(z2), A2)
                A2_nTA2z2 = self.unstride(F.conv_transpose2d(self.unzpad(A2z2),
                                                             A2_n, stride=2))

                out -= 2 * A2_nTA2z2

                A2_nz1 = F.conv2d(self.cpad(z1), A2_n, stride=2)
                A2_nTA2_nz1 = self.unstride(F.conv_transpose2d(A2_nz1,
                                                               A2_n, stride=2))

                out -= A2_nTA2_nz1

            return out

        n = len(self.A0)
        g_out = []
        for i in range(n - 1):
            g_out.append(multiply_zi(g[i], self.A(i), self.B(i), z2=g[i + 1], A2_n=self.A_n(i), A2=self.A(i + 1)))
        g_out.append(multiply_zi(g[n - 1], self.A(n - 1), self.B(n - 1)))

        return g_out

    def init_inverse(self, alpha, beta):
        n = len(self.A0)
        conv_fft_A = [init_fft_conv(self.A(i), (self.conv_shp[i], self.conv_shp[i]))
                      for i in range(n)]
        conv_fft_B = [init_fft_conv(self.B(i), (self.conv_shp[i], self.conv_shp[i]))
                      for i in range(n)]

        conv_fft_A_n = [init_fft_conv(self.A_n(i - 1), (self.conv_shp[i - 1], self.conv_shp[i - 1]))
                        for i in range(1, n)]

        I = [torch.eye(2 * self.A0[i].weight.shape[1], dtype=self.A0[i].weight.dtype,
                       device=self.A0[i].weight.device)[None, :, :] for i in range(n)]

        D1 = [(alpha + beta - beta * self.m) * I[i] \
              - beta * conv_fft_A[i].transpose(1, 2) @ conv_fft_A[i] \
              + beta * conv_fft_B[i] - beta * conv_fft_B[i].transpose(1, 2)
              for i in range(n - 1)]

        self.D1inv = [torch.inverse(D) for D in D1]

        self.D2 = [np.sqrt(-beta) * conv_fft_A_n[i] for i in range(n - 1)]

        G = [(self.D2[i] @ self.D1inv[i] @ self.D2[i].transpose(1, 2))[self.S_idx[i]] for i in range(n - 1)]
        S = [G[i][:self.conv_shp[i] ** 2 // 4]
             + G[i][self.conv_shp[i] ** 2 // 4:self.conv_shp[i] ** 2 // 2]
             + G[i][self.conv_shp[i] ** 2 // 2:3 * self.conv_shp[i] ** 2 // 4]
             + G[i][3 * self.conv_shp[i] ** 2 // 4:]
             for i in range(n - 1)]
        Hinv = [torch.eye(s.shape[1], device=s.device) + 0.25 * s for s in S]
        self.H = [torch.inverse(hinv).float() for hinv in Hinv]

        Wn = (1 - self.m) * I[n - 1] \
             - conv_fft_A[n - 1].transpose(1, 2) @ conv_fft_A[n - 1] \
             + conv_fft_B[n - 1] - conv_fft_B[n - 1].transpose(1, 2)

        self.Wn_inv = torch.inverse(alpha * I[n - 1] + beta * Wn)

        self.beta = beta

    def apply_inverse_conv(self, z, i):
        z0_fft = fft_to_complex_vector(torch.rfft(z, 2, onesided=False))
        y0 = 0.5 * z0_fft.bmm((self.D2[i] @ self.D1inv[i]).transpose(1, 2))[self.S_idx[i]]
        n = self.conv_shp[i]
        y1 = y0[:n ** 2 // 4] + y0[n ** 2 // 4:n ** 2 // 2] + y0[n ** 2 // 2:3 * n ** 2 // 4] + y0[3 * n ** 2 // 4:]
        y2 = y1.bmm(self.H[i].transpose(1, 2))
        y3 = y2.repeat(4, 1, 1)
        y4 = y3[self.S_idxT[i]]
        y5 = 0.5 * y4.bmm(self.D2[i] @ self.D1inv[i].transpose(1, 2))
        x0 = z0_fft.bmm(self.D1inv[i].transpose(1, 2)) - y5
        x0 = x0.view(n, n, x0.shape[1], -1, 2).permute(2, 3, 0, 1, 4)
        x0 = torch.irfft(x0, 2, onesided=False)
        return x0

    def apply_inverse_conv_transpose(self, g, i):
        g0_fft = fft_to_complex_vector(torch.rfft(g, 2, onesided=False))
        y0 = 0.5 * g0_fft.bmm(self.D1inv[i] @ self.D2[i].transpose(1, 2))[self.S_idx[i]]
        n = self.conv_shp[i]
        y1 = y0[:n ** 2 // 4] + y0[n ** 2 // 4:n ** 2 // 2] + y0[n ** 2 // 2:3 * n ** 2 // 4] + y0[3 * n ** 2 // 4:]
        y2 = y1.bmm(self.H[i])
        y3 = y2.repeat(4, 1, 1)
        y4 = y3[self.S_idxT[i]]
        y5 = 0.5 * y4.bmm(self.D2[i] @ self.D1inv[i])
        x0 = g0_fft.bmm(self.D1inv[i]) - y5
        x0 = x0.view(n, n, x0.shape[1], -1, 2).permute(2, 3, 0, 1, 4)
        x0 = torch.irfft(x0, 2, onesided=False)
        return x0

    def inverse(self, *z):
        n = len(self.A0)
        x = [self.apply_inverse_conv(z[0], 0)]
        for i in range(n - 1):
            A_nx0 = self.zpad(F.conv2d(self.cpad(x[-1]), self.A_n(i), stride=2))
            ATA_nx0 = self.uncpad(F.conv_transpose2d(self.cpad(A_nx0), self.A(i + 1)))
            xn = -self.beta * 2 * ATA_nx0
            if i < n - 2:
                x.append(self.apply_inverse_conv(z[i + 1] - xn, i + 1))
            else:
                x.append(fft_conv(z[i + 1] - xn, self.Wn_inv))

        return tuple(x)

    def inverse_transpose(self, *g):
        n = len(self.A0)

        x = [fft_conv(g[-1], self.Wn_inv, transpose=True)]
        for i in range(n - 2, -1, -1):
            A2x2 = F.conv2d(self.cpad(x[-1]), self.A(i + 1))
            A2_NTA2x2 = self.unstride(F.conv_transpose2d(self.unzpad(A2x2),
                                                         self.A_n(i), stride=2))
            xp = -self.beta * 2 * A2_NTA2x2
            x.append(self.apply_inverse_conv_transpose(g[i] - xp, i))
        x.reverse()
        return tuple(x)


def getShapeConv(in_shape, conv_shape, stride = 1, padding = 0):
    inChan, inH, inW = in_shape
    outChan, kH, kW = conv_shape[:3]

    outH = 1 + int((2 * padding + inH - kH) / stride)
    outW = 1 + int((2 * padding + inW - kW) / stride)
    return (outChan, outH, outW)


class MONConv(nn.Module):
    """ MON class with a single convolution """

    def __init__(
            self,
            in_channels,
            out_channels,
            shp,
            kernel_size=3,
            lben=True,
            lben_cond=3,
            m=0.5,
            bn_U=True,
            U_act=True):
        super().__init__()
        self.bn_U = bn_U
        if bn_U:
            self.bn_U_module = nn.BatchNorm2d(out_channels)

        self.M = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
        self.m = m
        self.out_dim = out_channels
        self.shp = shp
        self.kernel_size = kernel_size
        assert (kernel_size % 2 == 1)  # for simplicity, use odd kernel size
        self.padding = (kernel_size - 1) // 2
        self.U = nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.padding, stride=1, bias=True)  # weight on x

        self.lben = lben
        self.lben_cond = lben_cond
        if self.lben:
            self.lben_p = nn.Parameter(torch.zeros((1, out_channels, 1, 1)))
            self.lben_min = 1.0 / self.lben_cond
            self.lben_scale = 1 - self.lben_min

        self.norm_names = ['U_2', 'M_2', 'U_1', 'M_1']

        self.U_act = U_act

        if self.U_act:
            self.U_post_act_bias = nn.Parameter(torch.zeros((1, out_channels, 1, 1)))

        bias_layers = [self.U]
        if self.bn_U:
            bias_layers += [self.bn_U_module]
        if self.U_act:
            bias_layers += [nn.ReLU()]
        if self.U_act:
            bias_layers += [Bias(bias=self.U_post_act_bias)]

        self.bias_layers = nn.Sequential(*bias_layers)
        self.bias_layers_abs = Sequential.from_concrete_network(self.bias_layers, (in_channels,*shp))
        self.bias_precomp = None

        assert (0 <= self.m and 1 >= self.m)

    def x_shape(self, n_batch):
        return ((n_batch, self.U.in_channels, self.shp[0], self.shp[1]),)

    def z_shape(self, n_batch):
        return ((n_batch, self.out_dim, self.shp[0], self.shp[1]),)

    def prep_model(self):
        diag = self._compute_diag(self.M.weight)
        self.diag = torch.nn.Parameter(torch.tensor(diag), requires_grad=False)

    def forward(self, x, z, update_bn=True):
        a = self.bias(x)
        b = self.multiply(z)
        return (a+b,)

    def reset(self):
        self.bias_precomp = None
        if self.bias_layers_abs is not None:
            self.bias_layers_abs.reset_bounds()

    def _forward_abs(self, x, z):
        if self.bias_precomp is None:
            self.bias_precomp = self.bias_abs(x).detach()
        a = self.bias_precomp
        d = self.diag
        sqrt_d = 1.0 / torch.sqrt(d)

        if self.lben:
            lben_scaling = self.lben_scale * F.sigmoid(self.lben_p) + self.lben_min
            scaling_to_use = lben_scaling
            z = [scaling_to_use * zi for zi in z]

        out = sqrt_d * (sqrt_d * z).conv_transpose2d(self.M.weight, None, stride=1, padding=self.padding, output_padding=0, groups=1, dilation=1)
        b = out if not self.lben else (1.0 / scaling_to_use * out)

        return a+b

    def bias(self, x):
        return self.bias_layers(x)

    def bias_abs(self, x):
        return self.bias_layers_abs(x)

    def _row_col_sums(self, M):
        abs_M = torch.abs(M)
        one_vec = torch.ones((1, self.out_dim, self.shp[0], self.shp[1]),
                             dtype=abs_M.dtype, device=abs_M.device)
        M1 = F.conv2d(one_vec, abs_M, padding=self.padding)
        M1T = F.conv_transpose2d(one_vec, abs_M, padding=self.padding)
        return (M1 + M1T) / 2

    def _compute_diag(self, M):
        return self._row_col_sums(M) / (1 - self.m) + 1e-8

    def _multiply(self, M, *z, transpose=False):
        d = self.diag
        sqrt_d = 1.0 / torch.sqrt(d)
        conv_func = F.conv2d if transpose else F.conv_transpose2d

        if self.lben:
            lben_scaling = self.lben_scale * F.sigmoid(self.lben_p) + self.lben_min
            scaling_to_use = lben_scaling if not transpose else 1.0 / lben_scaling
            z = [scaling_to_use * zi for zi in z]

        out = sqrt_d * conv_func(sqrt_d * z[0], M, stride=1, padding=self.padding)
        return out if not self.lben else (1.0 / scaling_to_use) * out

    def multiply(self, *z):
        return self._multiply(self.M.weight, *z, transpose=False)

    def multiply_transpose(self, *g):
        return self._multiply(self.M.weight, *g, transpose=True)

    def get_norms(self):
        with torch.no_grad():
            U_2 = torch.norm(self.U.M.weight.data).detach().item()
            U_1 = torch.sum(torch.abs(self.U.M.weight.data)).item()
            M_2 = torch.norm(self.M.weight.data).detach().item()
            M_1 = torch.sum(torch.abs(self.M.weight.data)).detach().item()

        return {
            'U_2': U_2,
            'U_1': U_1,
            'M_2': M_2,
            'M_1': M_1
        }

    def params_for_ibp_init(self):
        return self.U.named_parameters()