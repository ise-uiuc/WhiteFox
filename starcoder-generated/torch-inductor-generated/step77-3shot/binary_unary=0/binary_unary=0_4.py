
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=2, dilation=2)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=2, dilation=2)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=2, dilation=2)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        a1 = v1.mean()
        v2 = self.conv2(x2)
        a2 = v2.mean()
        v3 = self.conv3(x3)
        a3 = v3.mean()
        return a1 + a2 + a3
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
