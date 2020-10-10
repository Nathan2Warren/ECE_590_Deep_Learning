import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from pruned_layers import *

def summary(net, printme = True):
    num_nz = []
    assert isinstance(net, nn.Module)
    if printme == True:
        print("Layer id\tType\t\tParameter\tNon-zero parameter\tSparsity(\%)")
    layer_id = 0
    num_total_params = 0
    num_total_nonzero_params = 0
    for n, m in net.named_modules():
        if isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy()
            weight = weight.flatten()
            num_parameters = weight.shape[0]
            num_nonzero_parameters = (weight != 0).sum()
            num_nz.append(num_nonzero_parameters)
            sparisty = 1 - num_nonzero_parameters / num_parameters
            layer_id += 1
            if printme == True: 
                print("%d\t\tLinear\t\t%d\t\t%d\t\t\t%f" %(layer_id, num_parameters, num_nonzero_parameters, sparisty)) 
            num_total_params += num_parameters
            num_total_nonzero_params += num_nonzero_parameters
        elif isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            weight = weight.flatten()
            num_parameters = weight.shape[0]
            num_nonzero_parameters = (weight != 0).sum()
            num_nz.append(num_nonzero_parameters)
            sparisty = 1 - num_nonzero_parameters / num_parameters
            layer_id += 1
            if printme == True: 
                print("%d\t\tConvolutional\t%d\t\t%d\t\t\t%f" % (layer_id, num_parameters, num_nonzero_parameters, sparisty)) 
            num_total_params += num_parameters
            num_total_nonzero_params += num_nonzero_parameters
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            layer_id += 1
            if printme == True:
                print("%d\t\tBatchNorm\tN/A\t\tN/A\t\t\tN/A" % (layer_id))
        elif isinstance(m, nn.ReLU):
            layer_id += 1
            if printme ==True:
                print("%d\t\tReLU\t\tN/A\t\tN/A\t\t\tN/A" % (layer_id))

    print("Total nonzero parameters: %d" %num_total_nonzero_params)
    print("Total parameters: %d" %num_total_params)
    total_sparisty = 1. - num_total_nonzero_params / num_total_params
    print("Total sparsity: %f" %total_sparisty)
    
    return total_sparisty, num_nz

