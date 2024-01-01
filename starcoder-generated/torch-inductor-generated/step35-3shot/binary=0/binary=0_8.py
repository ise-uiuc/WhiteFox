
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1, dilation=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 4, stride=2, padding=1, dilation=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v3 = x1 + other
        v3 = self.conv2(v3)
        v4 = v1 + v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 8, 8)
