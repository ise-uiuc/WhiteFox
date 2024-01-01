
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(12)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = 3 + t1
        t3 = torch.clamp_min(t2, 0)
        t4 = torch.clamp_max(t3, 6)
        t5 = t1 * t4
        t6 = t5 / 6
        t7 = t6.view((x1.shape[0], -1))
        t8 = self.bn(t7)
        t9 = t8.unsqueeze(-1).unsqueeze(-1)
        return t9 * t1
# Inputs to the model
x1 = torch.randn(15, 3, 256, 256)
