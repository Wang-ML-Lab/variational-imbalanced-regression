import torch
import torch.nn.functional as F
from evidential_deep_learning import *
import logging

# print = logging.info


def get_normal_output(inputs, is_train=False, weights=None, **kwargs):
    x_bar, n, omega = torch.tensor_split(inputs, 3, dim=1)
    
    flag_tensor = torch.ones_like(n)
    
    if is_train:
        n = n*weights
        
    gamma_v_prior = kwargs['gamma']*kwargs['v'] * flag_tensor
    
        
    v = kwargs['v'] * flag_tensor + n
    alpha = kwargs['alpha'] * flag_tensor + n / 2.0
    # if alpha.min().sum() > 0:
    #     alpha += kwargs['alpha'] * flag_tensor
    gamma = x_bar
    beta = omega
    
    return gamma, v, alpha, beta


def cdm_loss(inputs, targets, coeff=1.0, is_train=False, weights=None, **kwargs):
    gamma, v, alpha, beta = get_normal_output(inputs, is_train, weights, **kwargs)
    
    ''' 
    print('==')
    print(v.mean())
    print(alpha.mean())
    print(beta.mean())
    print((beta/(v*(alpha-1))).mean())
    '''

    loss_nll = NIG_NLL(targets, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(targets, gamma, v, alpha, beta, cdm=True)
    
    loss = loss_nll + coeff * loss_reg
    # if weights is not None:
    #     loss_nll *= weights.expand_as(loss_nll)
    # loss = loss_nll + coeff * loss_reg
    loss = torch.mean(loss)
    
    return loss


def weighted_edl_loss(inputs, targets, coeff=1.0, weights=None):
    
    gamma, v, alpha, beta = torch.tensor_split(inputs, 4, dim=1)
    
    '''
    print('==')
    print(v.mean())
    print(alpha.mean())
    print(beta.mean())
    print((beta/(v*(alpha-1))).mean())
    '''

    loss_nll = NIG_NLL(targets, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(targets, gamma, v, alpha, beta)

    loss = loss_nll + coeff * loss_reg
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, beta=1., weights=None):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss
