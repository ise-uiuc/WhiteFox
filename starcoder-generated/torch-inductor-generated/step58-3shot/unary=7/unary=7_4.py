s
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * 0.016065306594849
        v3 = v1 * torch.clamp(min=0, max=6.0, v1 + 3.0)
        v4 = v3 / 6.0
        return v4

class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
m1 = Model1()
m2 = Model2()

# Inputs to the model
x1 = torch.randn(1, 16)
__output1__ = m1(x1)
__output2__ = m2(x1)
