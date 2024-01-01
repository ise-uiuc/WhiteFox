
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(in_channels=3, out_channels=8)
 
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return torch.cat([v6, x2], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 8, 64, 64)
