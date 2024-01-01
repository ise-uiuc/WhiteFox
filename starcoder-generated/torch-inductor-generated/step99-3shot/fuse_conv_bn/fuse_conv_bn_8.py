
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(5)
        self.bn2 = torch.nn.BatchNorm2d(5)
        self.act1 = torch.nn.GELU()
        self.act2 = torch.nn.Sigmoid()
    def forward(self, x2):
        x1 = self.conv(x2)
        x1 = self.bn1(x1)
        x2 = self.act1(x1)
        x2, x3 = x2.chunk(2, dim=1)
        x2 = self.act2(x2)
        x2 = self.bn2(x2)
        y3 = torch.cat([x3, x2], dim = 1)
        return y3
# Inputs to the model
x2 = torch.randn(1, 6, 32, 32)
