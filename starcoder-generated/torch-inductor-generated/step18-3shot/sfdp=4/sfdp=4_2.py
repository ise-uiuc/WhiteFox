
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2, x3):
        q = self.conv(x1)
        k = self.conv(x2)
        v = self.conv(x3)
        v2 = v * 0.5
        v3 = v * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
v3 = v * 0.7071067811865476
x3 = torch.randn(1, 3, 64, 64)
