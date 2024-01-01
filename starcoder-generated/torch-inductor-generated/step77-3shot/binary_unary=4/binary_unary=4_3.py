
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1).clamp(min=0.0, max=20.0)
        v2 = self.other + v1
        v3 = v2.clamp(min=0.0, max=16.0)
        return v3

# Initializing the model
m = Model(other=torch.randn(1, 4) * 100)

# Inputs to the model
x1 = torch.randn(1, 4)
