
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.flatten = torch.nn.Flatten()
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.flatten(v1)
        v3 = v2 + x2
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 8, 8)
x2 = torch.randn(1, 32, 1, 1)
