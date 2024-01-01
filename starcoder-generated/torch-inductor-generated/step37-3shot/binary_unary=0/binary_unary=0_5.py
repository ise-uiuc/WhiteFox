
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1, stride=1)
    def forward(self, x, y, z):
        v1 = self.conv(x)
        v2 = v1 + y
        v3 = z + v2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
y = torch.randn(1, 16, 64, 64)
z = torch.randn(1, 16, 64, 64)
