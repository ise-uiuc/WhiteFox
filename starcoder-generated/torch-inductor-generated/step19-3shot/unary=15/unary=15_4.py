
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3,5,5, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(5)
    def forward(self,x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
