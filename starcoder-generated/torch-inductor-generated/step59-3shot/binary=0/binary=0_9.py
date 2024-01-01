
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(2)
    def forward(self, x1, other=2):
        v1 = self.conv(x1)
        if other == 2:
            other = torch.randn(v1.shape)
        v2 = self.bn(v1)
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
