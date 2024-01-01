
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = torch.nn.ModuleList([torch.nn.Conv2d(3, 16, 3, 1, 1), torch.nn.Sequential(*[torch.nn.Conv2d(16, 16, 3, 1, 1), torch.nn.BatchNorm2d(16)])])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return concatenated_tensor
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Block()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return concatenated_tensor
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
