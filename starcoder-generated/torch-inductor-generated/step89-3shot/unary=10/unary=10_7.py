
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = v1 * 0.5
        v2 = v1 * 0.7071067811865476
        v3 = torch.erf(v2)
        v3 = v3 + 1
        v3 = v3 * 6
        v3 = torch.clamp_min(v3, 0)
        v3 = torch.clamp_max(v3, 6)
        v3 = v3 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
