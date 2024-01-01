
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv = torch.nn.Conv2d(16, 32, 3, padding=1)
    def forward(self, x3):
        y1 = self.conv(x3)
        t1 = self.bn(y1)
        t2 = self.conv(t1)
        return None
# Inputs to the model
x3 = torch.randn(2, 16, 10, 10)
