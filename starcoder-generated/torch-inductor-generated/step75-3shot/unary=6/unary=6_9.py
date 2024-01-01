
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_0 = torch.nn.modules.hardtanh.Hardtanh(0, 6, False)
        self.module_1 = torch.nn.modules.hardtanh.Hardtanh(0, 6, True)
    def forward(self, x2):
        v1 = self.module_1(x2)
        v2 = self.module_0(v1)
        return v2
# Inputs to the model
x2 = torch.randn(10, 4, 4)
