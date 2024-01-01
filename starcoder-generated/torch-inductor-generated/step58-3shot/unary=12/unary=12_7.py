
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, dilation=2, padding=4)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=1, dilation=2, padding=4)
        self.conv3 = torch.nn.Conv2d(64, 1, 3, stride=1, dilation=2, padding=4)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.sigmoid(v3)
        v5 = v1 * v3
        v6 = v5 * v4
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
