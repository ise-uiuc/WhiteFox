
# Please fill in the blanks below.
import torch

class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    t1 = torch.nn.functional.adaptive_avg_pool2d(x, ___)
    t2 = torch.nn.functional.adaptive_avg_pool2d(t1, ___)
    v = t1 - t2
    return v
# Input tensor
x = torch.randn(1, 3, 32, 32)
