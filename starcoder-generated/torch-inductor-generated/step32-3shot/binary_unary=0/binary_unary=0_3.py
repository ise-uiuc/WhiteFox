
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.maxpool = torch.nn.MaxPool2d(4, stride=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.maxpool(v1)
        v3 = v2 - v2
        v4 = v3 / v3
        return v4
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
