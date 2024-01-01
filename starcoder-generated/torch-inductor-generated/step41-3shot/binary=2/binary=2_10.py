
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(3, affine=False)
    def forward(self, x):
        v0 = self.bn(x)
        v1 = self.conv1(v0)
        v2 = v1 - 1.3
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
