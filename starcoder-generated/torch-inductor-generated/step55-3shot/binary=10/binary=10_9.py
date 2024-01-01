
class OtherInfo:
    def __init__(self):
        self.other = torch.nn.Parameter(-torch.ones(16, 1000, dtype=torch.float32))

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 200, bias=False)
        self.other = OtherInfo()
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other.other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100)
