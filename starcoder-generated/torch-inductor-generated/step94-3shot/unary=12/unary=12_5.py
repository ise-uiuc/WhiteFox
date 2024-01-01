
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 1, stride=1, padding=1, dilation=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv1 = torch.nn.Conv2d(4, 1, 1, stride=1, padding=0, dilation=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.sigmoid(v3)
        v5 = self.conv1(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 4, 32)
