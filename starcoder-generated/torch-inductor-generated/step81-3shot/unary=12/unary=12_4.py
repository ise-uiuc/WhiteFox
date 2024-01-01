
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 16, 3, stride=1, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 1, stride=1, padding=0, dilation=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv2(v3)
        v5 = torch.sigmoid(v4)
        return v3, v5
# Inputs to the model
x1 = torch.randn(1, 32, 17, 17)
