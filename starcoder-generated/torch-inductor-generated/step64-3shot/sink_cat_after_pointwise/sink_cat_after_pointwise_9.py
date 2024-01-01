
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([torch.flatten(torch.cat((x, x), dim=1)).view(-1, 4, 4), x], dim=1)
        z = torch.cat((y, y, y), dim=1) if y.shape!= (64, 6, 4, 4) else y.view(y.shape[0], -1)
        return z
# Inputs to the model
x = torch.randn(2, 3, 4, 4)
