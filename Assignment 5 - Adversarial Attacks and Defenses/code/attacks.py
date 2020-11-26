import torch
import torch.nn as nn
import torch.nn.functional as F

# Compute the gradient of the loss w.r.t. the input data
def gradient_wrt_data(model,device,data,lbl):
    dat = data.clone().detach()
    dat.requires_grad = True
    out = model(dat)
    loss = F.cross_entropy(out,lbl)
    model.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()

def PGD_attack(model, device, dat, lbl, eps, alpha, iters, rand_start):
    # TODO: Implement the PGD attack
    # - dat and lbl are tensors
    # - eps and alpha are floats
    # - iters is an integer
    # - rand_start is a bool
    return 0

def FGSM_attack(model, device, dat, lbl, eps):
    # TODO: Implement the FGSM attack
    # - Dat and lbl are tensors
    # - eps is a float
    return 0

def MomentumIterative_attack(model, device, dat, lbl, eps, alpha, iters, mu):
    # TODO: Implement the Momentum Iterative Method
    # - dat and lbl are tensors
    # - eps, alpha and mu are floats
    # - iters is an integer
    return 0