import torch
import torch.nn as nn
import numpy as np
from random import sample

gpu_ind= 0
device = torch.device('cuda', gpu_ind)
print("Training G via GPU computing starts!")
n=300
S= 100
n_b = int(n/S)
sub_size = S
K0 = 100
K = 10

