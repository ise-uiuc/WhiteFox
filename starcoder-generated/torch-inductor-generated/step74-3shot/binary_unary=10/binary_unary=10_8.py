
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = torch.nn.Linear(16, 8)
 
    def forward(self, x1):
        v1 = self.l0(x1)
        v2 = v1 + x1
        v3 = torch.nn.functional.relu(v2)
        return v3

## Description of requirement
`t2 = t1 + other` requires `other` is a tensor.

# Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = torch.nn.Linear(16, 8)
 
    def forward(self, x1):
        v1 = self.l0(x1)
        v2 = torch.add(v1, 0.5)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
