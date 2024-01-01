
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential()
        self.features.add_module('0', torch.nn.Conv2d(3, 32, 3, 1, bias=True))
        self.features.add_module('1', torch.nn.Linear(32, 64))
        self.features.add_module('2', torch.nn.ReLU(inplace=False))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
