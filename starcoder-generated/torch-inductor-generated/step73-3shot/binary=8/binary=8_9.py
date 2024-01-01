
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 9, 3, stride=(1, 2), padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(9)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.bn(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 33, 33)
