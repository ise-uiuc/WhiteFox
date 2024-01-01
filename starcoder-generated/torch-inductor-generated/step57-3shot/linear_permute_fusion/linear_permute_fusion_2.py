
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2).to(torch.float64)
    def forward(self, x1):
        v4 = x1
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v2 = v1.permute(1, 2, 0)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
# Model begins

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x1):
        v4 = x1
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v2 = v1.permute(1, 0, 2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
# Model begins

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3).to(torch.float64)
    def forward(self, x1):
        self.training = True
        v4 = x1
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v2 = torch.transpose(v1, 0, 1)
        return torch.transpose(v2, 0, 1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
