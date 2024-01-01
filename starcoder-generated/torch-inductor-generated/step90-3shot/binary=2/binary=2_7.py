
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 6, 3, stride=3, padding=2, dilation=2, groups=4)
    def forward(self, x):
        t1 = self.conv(x)
        t2 = t1 - -33
        return t2
# Inputs to the model
x = torch.randn(1, 4, 24, 25)
