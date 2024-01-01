
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(128)
        self.conv = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.bn_1 = torch.nn.BatchNorm2d(128)
        self.act = torch.nn.ELU()
    def forward(self, x1):
        v1 = self.bn(x1)
        v2 = self.conv(v1)
        v3 = self.bn_1(v2)
        v4 = self.act(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 128, 172, 192)
