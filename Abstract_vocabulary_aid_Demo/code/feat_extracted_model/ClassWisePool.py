from __future__ import print_function 
from __future__ import division

import torch
import torch.nn as nn
import sys
from torch.autograd import Function

class ClassWisePoolFunction(Function):
    def __init__(self, num_maps):
        super(ClassWisePoolFunction, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        # batch dimension
        batch_size, num_channels, h, w = input.size()

        if num_channels % self.num_maps != 0:
            print('Error in ClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)

        num_outputs = int(num_channels / self.num_maps)
        x = input.view(batch_size, num_outputs, self.num_maps, h, w)
        output = torch.sum(x, 2)
        self.save_for_backward(input)
        return output.view(batch_size, num_outputs, h, w) / self.num_maps

    def backward(self, grad_output):
        input, = self.saved_tensors
        # batch dimension
        batch_size, num_channels, h, w = input.size()
        num_outputs = grad_output.size(1)

        grad_input = grad_output.view(batch_size, num_outputs, 1, h, w).expand(batch_size, num_outputs, self.num_maps,
                                                                               h, w).contiguous()
        return grad_input.view(batch_size, num_channels, h, w)
        
class ClassWisePool(nn.Module):
    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps=num_maps

    def forward(self, input):
         # batch dimension
        batch_size, num_channels, h, w = input.size()

        if num_channels % self.num_maps != 0:
            print('Error in ClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)

        num_outputs = int(num_channels / self.num_maps)
        x = input.view(batch_size, num_outputs, self.num_maps, h, w)
        output = torch.sum(x, 2)
        #self.save_for_backward(input)
        return output.view(batch_size, num_outputs, h, w) / self.num_maps


