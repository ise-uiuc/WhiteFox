
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.pool = torch.nn.AvgPool2d([2, 2], stride=2)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.flatten(v1, 1)
        v3 = torch.relu(v2)
        v4 = self.pool(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
