
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=3, dilation=1, groups=3)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        x1 = self.conv1(x1)
        x2 = x1.sigmoid()
        x3 = x2.mul(2)
        x4 = x3.tanh()
        return x4
# Inputs to the model
x1 = torch.randn(1, 3, 200, 200)
