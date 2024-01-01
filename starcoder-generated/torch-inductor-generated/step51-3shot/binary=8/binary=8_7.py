
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=3, padding=3, groups=3, bias=True)
        self.bn1 = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.bn1(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
