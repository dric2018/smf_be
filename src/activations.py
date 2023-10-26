"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 24 Oct, 2023
"""

import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
