
import torch
class Model(torch.nn.Module):
    def forward(self, x):
        x1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)(x)
        z = torch.add(x1, 3)
        w = torch.clamp(z, min=0, max=6)
        y = torch.div(w, 6)
        return y
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
