
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(2, 2, 1)
        torch.manual_seed(1)
        self.bn1 = torch.nn.BatchNorm2d(2)
        torch.manual_seed(1)
        self.bn2 = torch.nn.BatchNorm2d(2, affine=False)
    def forward(self, x2):
        v3 = self.bn1(self.conv(x2))
        v3 = self.conv(v3)
        v4 = self.conv(v3)
        v4a = self.bn1(v4)
        v4b = self.bn2(v4a)
        v4a = self.conv(v4b)
        return v4b
# Inputs to the model
x2 = torch.randn(1, 2, 3, 3)
