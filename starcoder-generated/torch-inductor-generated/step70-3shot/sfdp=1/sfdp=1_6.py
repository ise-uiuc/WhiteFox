
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(4, 4)
        self.key = nn.Linear(4, 4)
        self.value = nn.Linear(4, 4) 
 
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        scaled_qk = torch.matmul(q, k.transpose(0, 1))
        softmax_qk = F.softmax(scaled_qk, dim=-1)
        output = torch.matmul(softmax_qk, v)
        return output


# Initializing the model
m = Model()
  
# Inputs to the model
x1 = torch.randn(2, 4)
x2 = torch.randn(2, 4)
