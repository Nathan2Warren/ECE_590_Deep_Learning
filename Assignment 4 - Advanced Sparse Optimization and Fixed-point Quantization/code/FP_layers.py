import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, bit):  # "w" is the weight tensor (as a PyTorch Tensor), "bit" is the precision to be quantized to. You can ignore "ctx" as it will not be used
        if bit is None:
            wq = w         # Full precision, no change
        elif bit==0:
            wq = w*0       # Zero precision, everything is zero 
        else:
            # Your code here:
            alpha = torch.max(w) - torch.min(w) # Compute alpha (scale) for dynamic scaling
            beta = torch.min(w)         # Compute beta (bias) for dynamic scaling
            ws = (w - beta)/alpha   # Scale w with alpha and beta so that all elements in ws are between 0 and 1
            
            R = 1/(2**bit -1) * torch.round((2**bit-1)* ws)        # Quantize ws with a linear quantizer to "bit" bits
            
            wq =  alpha * R + beta         # Scale the quantized weight R back with alpha and beta
            # End of your code, do not change below this line
        return wq.to(device)

    @staticmethod
    def backward(ctx, g):
        return g, None

class FP_Linear(nn.Module):
    def __init__(self, in_features, out_features, Nbits=None):
        super(FP_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.Nbits = Nbits
        
        # Initailization
        m = self.in_features
        n = self.out_features
        self.linear.weight.data.normal_(0, math.sqrt(2. / (m+n)))

    def forward(self, x):
        return F.linear(x, STE.apply(self.linear.weight, self.Nbits), self.linear.bias)

    

class FP_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, Nbits=None):
        super(FP_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.Nbits = Nbits

        # Initialization
        n = self.kernel_size * self.kernel_size * self.out_channels
        m = self.kernel_size * self.kernel_size * self.in_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / (n+m) ))
        self.sparsity = 1.0

    def forward(self, x):
        return F.conv2d(x, STE.apply(self.conv.weight, self.Nbits), self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)

    



