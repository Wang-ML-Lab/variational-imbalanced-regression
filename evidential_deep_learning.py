import os,sys
import time, glob, argparse, random
import pandas as pd
import seaborn as sns
import scipy
import cv2
import json
from zipfile import ZipFile
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, roc_auc_score
from copy import deepcopy

from tqdm.notebook import tqdm

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image


class ConjugatePrior(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConjugatePrior, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, 3 * out_channels)
        self.softplus = nn.Softplus(beta=0.1)
        self.split_size = out_channels
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        output = self.linear(x)
        x_bar, n, beta = torch.tensor_split(output, 3, dim=1)
        n = self.softplus(n) + 2
#         x_bar = self.softplus(x_bar)
        beta = self.softplus(beta)
        return torch.cat((x_bar, n, beta), dim=1)
        

class Conv2DNormal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(Conv2DNormal, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels=2*out_channels, kernel_size=kernel_size, **kwargs)
        self.softplus = nn.Softplus()
        self.split_size = out_channels

    def forward(self, x):
        output = self.conv(x)
        mu, logsigma = torch.tensor_split(output, 2, dim=1)
        sigma = self.softplus(logsigma) + 1e-6
        return torch.cat((mu, sigma), dim=1)

class Conv2DNormalGamma(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(Conv2DNormalGamma, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels=4*out_channels, kernel_size=kernel_size, **kwargs)
        self.softplus = nn.Softplus()
        self.split_size = out_channels

    def forward(self, x):
        output = self.conv(x)
        mu, logv, logalpha, logbeta = torch.tensor_split(output, 4, dim=1)
        v = self.softplus(logv)
        alpha = self.softplus(logalpha) + 1
        beta = self.softplus(logbeta)
        return torch.cat((mu, v, alpha, beta), dim=1)


class LinearNormal(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(LinearNormal, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, 2 * out_channels)
        self.softplus = nn.Softplus()
        self.split_size = out_channels

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        output = self.linear(x)
        mu, logsigma = torch.tensor_split(output, 2, dim=1)
        sigma = self.softplus(logsigma) + 1e-6
        return torch.cat((mu, sigma), dim=1)


class LinearNormalGamma(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(LinearNormalGamma, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, 4 * out_channels)
        self.softplus = nn.Softplus(beta=0.01)
        self.split_size = out_channels

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        output = self.linear(x)
        mu, logv, logalpha, logbeta = torch.tensor_split(output, 4, dim=1)
        v = self.softplus(logv)
        alpha = self.softplus(logalpha) + 1.0
        beta = self.softplus(logbeta)
        return torch.cat((mu, v, alpha, beta), dim=1)


class LinearDirichlet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(LinearDirichlet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        output = self.linear(x)
        evidence = torch.exp(output)
        alpha = evidence + 1
        prob = alpha / torch.sum(alpha, dim=1, keepdims=True)
        return torch.cat((alpha, prob), dim=1)


class LinearSigmoid(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(LinearSigmoid, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        output = self.linear(x)
        prob = torch.sigmoid(output)
        return [output, prob]


def Dirichlet_SOS(y, alpha, t):
    def KL(alpha):
        beta=torch.ones((1,alpha.shape[1]),dtype=torch.float32)
        S_alpha = torch.sum(alpha,dim=1,keepdims=True)
        S_beta = torch.sum(beta,dim=1,keepdims=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha),dim=1,keepdims=True)
        lnB_uni = torch.sum(torch.lgamma(beta),dim=1,keepdims=True) - torch.lgamma(S_beta)

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)

        kl = torch.sum((alpha - beta)*(dg1-dg0),dim=1,keepdims=True) + lnB + lnB_uni
        return kl

    S = torch.sum(alpha, dim=1, keepdims=True)
    evidence = alpha - 1
    m = alpha / S

    A = torch.sum((y-m)**2, dim=1, keepdims=True)
    B = torch.sum(alpha*(S-alpha)/(S*S*(S+1)), dim=1, keepdims=True)

    # annealing_coef = tf.minimum(1.0,tf.cast(global_step/annealing_step,tf.float32))
    alpha_hat = y + (1-y)*alpha
    C = KL(alpha_hat)

    C = torch.mean(C, dim=1)
    return torch.mean(A + B + C)

def Sigmoid_CE(y, y_logits):
    loss = self.BCEWithLogitsLoss(y, y_logits)
    return torch.mean(loss)


def MSE(y, y_):
    ax = list(range(1, len(y.shape)))

    mse = torch.mean((y-y_)**2, dim=ax)
    return mse

def RMSE(y, y_):
    rmse = torch.sqrt(torch.mean((y-y_)**2))
    return rmse

def Gaussian_NLL(y, mu, sigma):
    ax = list(range(1, len(y.shape)))

    logprob = -torch.log(sigma) - 0.5*torch.log(2*np.pi) - 0.5*((y-mu)/sigma)**2
    loss = torch.mean(-logprob, dim=ax)
    return loss

def Gaussian_NLL_logvar(y, mu, logvar):
    ax = list(range(1, len(y.shape)))

    log_liklihood = 0.5 * (
    -torch.exp(-logvar)*(mu-y)**2 - torch.log(2*torch.tensor([np.pi], dtype=logvar.dtype)) - logvar
    )
    loss = torch.mean(-log_liklihood, dim=ax)
    return loss

def NIG_NLL(y, gamma, v, alpha, beta):    
    twoBlambda = 2*beta*(1+v)

    nll = 0.5*torch.log(np.pi/(v))  \
    - alpha*torch.log(twoBlambda)  \
    + (alpha+0.5) * torch.log(v*(y-gamma)**2 + twoBlambda)  \
    + torch.lgamma(alpha)  \
    - torch.lgamma(alpha+0.5)
    
    return nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5*(a1-1)/b1 * (v2*torch.square(mu2-mu1))  \
    + 0.5*v2/v1  \
    - 0.5*torch.log(torch.abs(v2)/torch.abs(v1))  \
    - 0.5 + a2*torch.log(b1/b2)  \
    - (torch.lgamma(a1) - torch.lgamma(a2))  \
    + (a1 - a2)*torch.digamma(a1)  \
    - (b1 - b2)*a1/b1
    return KL

def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, kl=False, cdm=False):
    error = F.l1_loss(gamma, y, reduction="none")

    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        if cdm:
            # evi = v + 2 * alpha
            evi = 2 * v + alpha
        else:
            evi = 2*v+alpha
        reg = error*evi

    return reg

def EvidentialRegression(evidential_output, y_true, coeff=1.0, use_cdm=False):
    gamma, v, alpha, beta = torch.tensor_split(evidential_output, 4, dim=1)
#     print(gamma.mean())
#     print(v.mean())
#     print(alpha.mean())
#     print(beta.mean())
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta, cdm=False)
#     print('ha')
#     print(loss_nll.mean())
#     print(loss_reg.mean())
    loss = loss_nll + coeff * loss_reg
    return [loss.mean(), loss_nll.mean(), loss_reg.mean()]


# def KL_DIV(p_z, q_z, kl_coeff=0.01):
#     kl = q_z - p_z
#     kl = kl.mean()
#     kl *= kl_coeff
#     return kl

def KL_DIV(mu, log_var, kl_coeff=0.1):
    target_mu = torch.zeros_like(mu)
    target_var = (torch.ones_like(log_var))**2

    kl = 0.5 * torch.sum(target_var**(-1) * log_var.exp() + target_var**(-1) * (target_mu-mu)**2 - 1 + torch.log(target_var) - log_var, dim=1)
#     kl = 0.5 * torch.sum(log_var.exp() + mu**2 - 1 - log_var, dim=1)
    kl = kl.mean()
    kl *= kl_coeff
    
    return kl
