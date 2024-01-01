
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=2, dilation=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 2, stride=2, padding=1, dilation=3)
        self.conv3 = torch.nn.Conv2d(16, 32, 1, stride=2, padding=2, bias=False)
        self.conv4 = torch.nn.Conv2d(32, 64, 2, stride=1, padding=2, dilation=4, groups=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = torch.tanh(v1 + v2)
        v4 = self.conv3(v3)
        v5 = F.relu(v4)
        v6 = self.conv4(v5)
        v7 = F.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 128, 64)
