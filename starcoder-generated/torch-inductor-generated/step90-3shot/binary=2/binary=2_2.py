
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 1, 1, stride=1, padding=0, dilation=2, groups=1)
    def forward(self, x):
        t1 = self.conv(x)
        t2 = t1 - 1.5
        return t2
# Inputs to the model
x = torch.randn(1, 32, 16, 16)
