
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 2, 31, stride=1, padding=5, dilation=1)
        self.conv2 = torch.nn.Conv2d(8, 2, 31, stride=1, padding=5, dilation=2)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = torch.mul(v1, v2)
        v4 = self.sigmoid(v3)
        v5 = v1 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
