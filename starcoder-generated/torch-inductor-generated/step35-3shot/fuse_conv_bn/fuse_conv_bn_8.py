
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x2):
        b = self.conv(x2)
        if x2.min() > 0:
            e = self.bn(b)
            return e
        a = self.bn(torch.squeeze(b))
        return a
# Inputs to the model
x2 = torch.randn(1, 3, 6, 6)
