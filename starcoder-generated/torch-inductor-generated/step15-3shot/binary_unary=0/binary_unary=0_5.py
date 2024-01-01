
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.pool = torch.nn.AdaptiveAvgPool2d(7)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + v1
        v3 = torch.relu(v2)
        v4 = self.pool(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
