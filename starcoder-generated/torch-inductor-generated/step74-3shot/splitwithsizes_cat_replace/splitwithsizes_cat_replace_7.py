
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1, 0, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
    def forward(self, v1):
        return (self.bn1(self.conv1(v1)),)
class Block1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
    def forward(self, v1, v2):
        split_tensors = torch.split(v2, [1], dim=1)
        concat, _ = self.block(v1)
        return (concat, split_tensors[0])
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block1()
        self.features = torch.nn.Sequential(self.block)
        self.extra = torch.nn.ReLU()
    def forward(self, v1, v2):
        split_tensors = torch.split(v2, [1], dim=1)
        concat, _ = self.block(v1, v2)
        return (concat, split_tensors[0])
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64, requires_grad=True)
x2 = torch.randn(1, 32, 64, 64, requires_grad=True)
