
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.avgpool = torch.nn.AvgPool2d(2, stride=2)
    def forward(self, x1, other):
        v1 = self.conv(x1)
        v2 = v1 + other
        v3 = self.avgpool(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
