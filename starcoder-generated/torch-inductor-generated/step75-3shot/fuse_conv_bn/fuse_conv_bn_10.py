
class ModuleBn0124(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(2, 4)
        self.bias = torch.randn(2)
    def forward(self, x):
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1) # x: [16, 2, 8, 8] weight: [2, 4] bias: [2]
        mean = x.mean((0, 2, 3), keepdim=True)         # mean: [2, 1, 1]
        var = x.var((0, 2, 3), keepdim=True)           # var: [2, 1, 1]
        var = torch.add(var, 0.001)                    # var += scalar
        x = (x - mean)/(var) * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        x = torch.add(x, x)
        return x

class ModuleBn2104(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.randn(1) # 1
        self.gamma = torch.randn(2) # 2
        self.eps = torch.zeros(2) # 0
        torch.manual_seed(1)
        self.weight = torch.randn(2, 4)
        torch.manual_seed(1)
    def forward(self, x):
        mean = x.mean((0, 2, 3), keepdim=True)
        var = x.var((0, 2, 3), keepdim=True)
        var = torch.add(var, self.eps)
        x = (x - mean)/(var.sqrt()) * self.weight.view(2, -1, 1, 1) + self.bias.view(1, -1, 1, 1) # 2 - 1 - 1
        gamma = self.gamma.view(2, -1, 1, 1)      # [2, 4, 1, 1]
        x = gamma * x                              # [2, 4, 8, 8] bias: [1, 1]
        return x

class ModuleBn11104(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(1)
        self.bias = torch.randn(1)
    def forward(self, x):
        mean = x.mean((0, 2, 3), keepdim=True)
        var = x.var((0, 2, 3), keepdim=True)
        var = torch.add(var, 0.001)
        x = (x - mean)/(var.sqrt()) * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1) # 2 - 1 - 1
        x = torch.concat([x, x], 2) # 0 axis
        x = torch.add(x, x)
        return x

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn0 = ModuleBn0124() #[None, 2, 8, 8]
        self.bn1 = ModuleBn11104() #[None, 4, 16, 16]
        self.bn2 = ModuleBn2104() #[None, 2, 8, 8]
    def forward(self, x1):
        x2 = self.bn0(x1)
        x3 = self.bn1(x2)
        x4 = self.bn2(x3)
        return torch.add(x4, x4)

# Inputs to the model
x1 = torch.randn(1, 2, 4, 4)
