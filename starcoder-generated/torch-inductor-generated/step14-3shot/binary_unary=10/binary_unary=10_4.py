
class Module1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + torch.rand(8)
        v3 = v2.relu()
        return v3

class Module2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module1 = Module1()
 
    def forward(self, x1):
        v1 = self.module1(x1)
        v2 = v1.relu()
        v3 = v1 + v2
        return v3

# Initializing the model
m = Module2()

# Inputs to the model
x1 = torch.randn(1, 8)
