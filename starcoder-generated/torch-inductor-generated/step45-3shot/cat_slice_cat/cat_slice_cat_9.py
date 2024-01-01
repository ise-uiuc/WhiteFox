
class Model(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.size = param
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:self.size]
        v3 = v2[:, 0:self.size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model(size=64)

# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
x2 = torch.randn(1, 32, 64, 64)
