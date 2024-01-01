
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 8, stride=1, padding=8, dilation=1, groups=1)
    def forward(self, x):
        t1 = self.conv(x)
        t2 = t1 - -3
        return t2
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
