
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 32, 7, stride=1, padding=3)
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.mul(v1, v2)
        v4 = self.conv1(x2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
