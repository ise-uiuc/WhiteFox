
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1, 0, bias=False, groups = 1)
        self.bn1 = torch.nn.BatchNorm2d(64)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        return torch.nn.ReLU()(self.bn1(self.conv1(torch.cat(split_tensors, dim=1, groups = 1))))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Block()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
