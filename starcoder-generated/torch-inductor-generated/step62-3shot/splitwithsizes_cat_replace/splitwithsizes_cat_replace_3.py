
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(23, 23))
    def forward(self, v1):
        return torch.nn.Linear(23, 23).cuda()(v1)
class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
    def forward(self, v1):
        return self.block(v1)
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Linear(42, 23)
    def forward(self, v1):
        return self.block(v1)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_0 = Model0()
        self.layer_1 = Model1()
    def forward(self, v1):
        return (self.layer_0(v1) * 2 + 10, torch.split(self.layer_1(v1), [2, 2], dim=1))
# Inputs to the model
x1 = torch.randn(1, 42).cuda()
