s
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 42.0
        v3 = torch.relu(v2)
        return v3

class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 80.0
        v3 = torch.relu(v2)
        return v3

# Models to inputs
x11 = torch.randn(1, 10)
x12 = torch.randn(1, 10)
__output1__ = Model1()(x11)
__output2__ = Model2()(x12)

