
class Model(torch.nn.Module):
    def __init__(self, negative_slope=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.negative_slope = negative_slope
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.relu1(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = v4 > 0
        v6 = v4 * self.negative_slope
        v7 = torch.where(v5, v4, v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
