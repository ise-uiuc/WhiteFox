
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool2d = torch.nn.Identity()
        self.linear = torch.nn.Linear(8192, 8192)
 
    def forward(self, x1):
        v1 = self.pool2d(x1)
        v2 = self.linear(v1)
        v3 = v2 + 3
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v5 / 6
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8192)
