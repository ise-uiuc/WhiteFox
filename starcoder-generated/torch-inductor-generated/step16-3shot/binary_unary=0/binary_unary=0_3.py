
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 7, stride=1, padding=3)
    def forward(self, x, y, z):
        v1 = self.conv(x)
        v2 = self.conv(v1)
        v3 = v2 + y
        v4 = torch.relu(v3)
        v5 = self.conv(v1)
        v6 = v5 + z
        return v4 + v6
# Inputs to the model
x = torch.randn(1, 2, 3, 3)
y = torch.randn(1, 3, 3, 3)
z = torch.randn(1, 1, 3, 3)
