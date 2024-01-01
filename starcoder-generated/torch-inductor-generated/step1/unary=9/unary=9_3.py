
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + 3
        v3 = torch.clamp(v2, 0, 6)
        v4 = torch.clamp(v2, 3, 7)
        v5 = v3 / 6
        v6 = v3 / v5
        v7 = v6 / 6
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
