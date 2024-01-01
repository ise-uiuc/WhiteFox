
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(8, 3, bias=False)
        self.scale = 6
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
 
        v7 = self.linear(v6)
        v8 = v7 + 3
        v9 = torch.clamp_min(v8, 0)
        v10 = torch.clamp_max(v9, self.scale)
        v11 = v10 / self.scale
        return v11
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
