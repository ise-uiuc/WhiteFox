
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=3, dilation=3)
        self.bn = torch.nn.BatchNorm2d(3)
        self.act = torch.nn.ReLU()
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = F.hardtanh(t2, min_val=0.0, max_val=6.0)
        t4 = t1 * t3
        t5 = t4 / 6
        t6 = self.bn(t5)
        t7 = self.act(t6)
        return t7
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
