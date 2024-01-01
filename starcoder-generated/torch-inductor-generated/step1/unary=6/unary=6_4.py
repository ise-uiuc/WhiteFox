
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + 3
        v3 = self.conv(v1)
        v4 = v3 + 1
        v5 = v4.clamp_min(0)
        v6 = v5.clamp_max(6)
        v7 = v2 * v6
        v8 = v7 / 6
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
