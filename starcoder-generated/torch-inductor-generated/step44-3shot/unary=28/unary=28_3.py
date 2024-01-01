
class Model(torch.nn.Module):
    def __init__(self, min=-1.0, max=1.0):
        super().__init__()
        self.linear = torch.nn.Linear(3, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, -1.0)
        v3 = torch.clamp_max(v2, 1.0)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
