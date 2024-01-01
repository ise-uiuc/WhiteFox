
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.mul_(0.5)
        v3 = v1.mul_(0.7071067811865476)
        v3 = torch.erf(v3)
        v4 = v3.add_(1)
        v5 = v2.mul_(v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
