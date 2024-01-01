
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 3, kernel_size=3, stride=2, padding=1, dilation=2, groups=3, bias=True)
        self.bn = torch.nn.BatchNorm1d(3, momentum=0.99)
    def forward(self, x1):
        s = self.conv(x1)
        t = self.bn(s)
        return t
# Inputs to the model
x1 = torch.randn(1, 3, 6)
