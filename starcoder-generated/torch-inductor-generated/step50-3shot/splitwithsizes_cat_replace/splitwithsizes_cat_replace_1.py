
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.MaxPool2d(3, 2, 2), torch.nn.Conv2d(32, 32, 3, 2, 3), torch.nn.AvgPool2d(3, 2, 2, ceil_mode=False)])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dim=3)
        concatenated_tensor = torch.cat(split_tensors, dim=3)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dim=3))
# Inputs to the model
x1 = torch.randn(1, 3, 8, 16)
