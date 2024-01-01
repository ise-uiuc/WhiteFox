
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        y1 = torch.zeros(3, 3).float()
        torch.manual_seed(1)
        y2 = torch.zeros(3, 3).float()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(1, 1, 1, bias=True)

        self.bn = torch.nn.BatchNorm2d(1, track_running_stats=True, affine=True)
    def forward(self, x1):
        s1 = self.conv(x1)
        s1 = self.bn(s1)
        s1 = self.conv(s1)
        s1 = self.bn(s1)
        return s1
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
