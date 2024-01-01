
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.linear = torch.nn.Linear(64, 16)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min)
        v3 = torch.clamp_max(v2, max)
        return v3

# Initializing the model
min = 0.04
max = 0.42
m = Model(min=min, max=max)

# Inputs to the model
x1 = torch.randn(1, 64)
