import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _huffman_coding_per_layer(weight, centers):
    """
    Huffman coding for each layer
    :param weight: weight parameter of the current layer.
    :param centers: KMeans centroids in the quantization codebook of the current weight layer.
    :return: encoding map and frequency map for the current weight layer.
    """
    # Your code: Implement the Huffman coding here. Store the encoding map into 'encoding'
    # and frequency map into 'frequency'.
    
    ### Frequency
    # Only looking at nonzero weights
    non_zero_idx = np.nonzero(weight)
    weight = weight[non_zero_idx]
    
    # Get a count of the unique weights 
    unique, counts = np.unique(weight, return_counts=True)
    
    # Calculate the frequency
    count_freq = np.zeros(len(counts))
    for i in range(len(counts)):
        count_freq[i] = counts[i]/len(weight)
        
    # Create dict of centers as index and 
    frequency = dict(zip(centers, count_freq))
    
    ### Encoding
    encodings = huffman_encoder(frequency)
    
    return encodings, frequency

def huffman_encoder(frequency):
    # Base case, we assign 0 and 1 for dictionary of length 2, will end with this
    if len(frequency) == 2:
        return dict(zip(frequency.keys(), ['0','1']))
    
    # Sort by value which is frequency
    freq_copy = frequency.copy()
    sorted_freq = sorted(freq_copy.items(), key = lambda p: p[1])
    
    # Select two smallest values
    low1 = sorted_freq[0][0]
    low2 = sorted_freq[1][0]
    
    # Pop off lowest frequency
    frq1 = freq_copy.pop(low1)
    frq2 = freq_copy.pop(low2)
    
    # Create new dictionary k-v pair which is combined lowest two frequency pairs
    freq_copy[low1 + low2] = frq1 + frq2
    
    # Recursion 
    encodings = huffman_encoder(freq_copy)
    # We remove 
    remove = encodings.pop(low1 + low2)
    encodings[low1] = remove + '0'
    encodings[low2] = remove + '1'
    
    return encodings

def compute_average_bits(encodings, frequency):
    """
    Compute the average storage bits of the current layer after Huffman Coding.
    :param encodings: encoding map of the current layer w.r.t. weight (centriod) values.
    :param frequency: frequency map of the current layer w.r.t. weight (centriod) values.
    :return (float) a floating value represents the average bits.
    """
    total = 0
    total_bits = 0
    for key in frequency.keys():
        total += frequency[key]
        total_bits += frequency[key] * len(encodings[key])
    return total_bits / total

def huffman_coding(net, centers):
    """
    Apply huffman coding on a 'quantized' model to save further computation cost.
    :param net: a 'nn.Module' network object.
    :param centers: KMeans centroids in the quantization codebook for Huffman coding.
    :return: frequency map and encoding map of the whole 'net' object.
    """
    assert isinstance(net, nn.Module)
    layer_ind = 0
    freq_map = []
    encodings_map = []
    huff_bits = []
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)
            huff_bits.append(huffman_avg_bits)
        elif isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)
            huff_bits.append(huffman_avg_bits)

    return freq_map, encodings_map, huff_bits