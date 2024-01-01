
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(10, 15, 3, stride=2, padding=1)
    def forward(self, x):
        y = self.conv1(x)
        v0 = self.conv2(y) - 1.4e+12
        return y, v0
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
