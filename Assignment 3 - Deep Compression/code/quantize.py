import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _quantize_layer(weight, bits=8):
    """
    :param weight: A numpy array of any shape.
    :param bits: quantization bits for weight sharing.
    :return quantized weights and centriods.
    """
    # Your code: Implement the quantization (weight sharing) here. Store 
    # the quantized weights into 'new_weight' and store kmeans centers into 'centers_'
    
    # Store shape for conversion back 
    weight_shape = weight.shape
    
    # Flatten 
    weight_flattened = weight.flatten()
    
    # Find index of nonzero weights
    non_zero_idx = [i for i, e in enumerate(weight_flattened) if e != 0]
    
    # Create array of weights that are non zero
    non_zero_weights = weight_flattened[non_zero_idx]
    
    # Use KMeans to find a value for weights
    k_mean = KMeans(n_clusters = 2**bits)
    
    # KMeans with linear initialization
    linear_space = np.linspace(non_zero_weights.min(), non_zero_weights.max(), 2**bits).reshape(-1,1)
    k_mean = KMeans(n_clusters = 2**bits, init = linear_space, n_init = 1) 
    
    # KMeans with random initialization
    #k_mean = KMeans(n_clusters = 2**bits, init = "random")
    
    k_mean.fit(non_zero_weights.reshape(-1,1))
    
    # Store cluster and label
    centers_ = k_mean.cluster_centers_
    labs = k_mean.labels_
    
    # Create zero array and then fill in weights that are updated by index 
    new_weight = np.zeros(weight_shape).reshape(-1,1)
    new_weight[non_zero_idx] = centers_[labs]
    new_weight = new_weight.reshape(weight_shape)

    return new_weight, centers_

def quantize_whole_model(net, bits=8):
    """
    Quantize the whole model.
    :param net: (object) network model.
    :return: centroids of each weight layer, used in the quantization codebook.
    """
    cluster_centers = []
    assert isinstance(net, nn.Module)
    layer_ind = 0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            weight, centers = _quantize_layer(weight, bits=bits)
            centers = centers.flatten()
            cluster_centers.append(centers)
            m.conv.weight.data = torch.from_numpy(weight).to(device)
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
        elif isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy()
            weight, centers = _quantize_layer(weight, bits=bits)
            centers = centers.flatten()
            cluster_centers.append(centers)
            m.linear.weight.data = torch.from_numpy(weight).to(device)
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
    return np.array(cluster_centers)

