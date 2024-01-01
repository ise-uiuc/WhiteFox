
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.ReLU(inplace=False))
    def forward(self, x3):
        return self.features(x3)
class Block1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 7, 1, 3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 64, 7, 1, 3, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    def forward(self, v0):
        split_tensors = torch.split(v0, [1, 1, 1], dim=1)
        concated_tensor = torch.cat(split_tensors, dim=1)
        return split_tensors[0]
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.nn.Sequential(Block(), Block1())
        self.extra = Block()
    def forward(self, v0):
        split_tensors = torch.split(v0, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, split_tensors[1])
# Inputs to the model
x0 = torch.randn(1, 3, 64, 64)
