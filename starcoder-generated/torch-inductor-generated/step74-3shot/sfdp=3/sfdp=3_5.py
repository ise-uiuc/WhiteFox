
from typing import Optional

import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.scale_factor = 1 / (dropout_p ** 2)
 
    def forward(self, x):
        qk = torch.matmul(x, x.t())
        scaled_qk = qk * self.scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(x)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 128, 100)
