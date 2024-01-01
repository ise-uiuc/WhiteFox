
class Block(torch.nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 3, 1, 1)
    def forward(self, v1):
        split0 = torch.split(v1, [1], dim=1)[0]
        return torch.nn.ReLU()(self.conv1(split0))
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.nn.Sequential(*[Block() for _ in range(1)])
    def forward(self, v1):
        split0 = torch.split(v1, [1], dim=1)[0]
        return split0
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
