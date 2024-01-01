
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        nn.ModuleList([nn.Tanh()]).cuda()
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.m = nn.Sequential(nn.Tanh())
# Inputs to the model
input = torch.randn(2, 3, 4)
