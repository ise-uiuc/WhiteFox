
import torch.nn.functional as F
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv = torch.nn.Conv2d(512, 32, 3, padding=1, stride=2)
    def forward(self, x0):
        c0 = self.conv(x0)
        c1 = F.tanh(c0)
        return c1
# Inputs to the model
x0 = torch.randn(4, 512, 16, 16)
