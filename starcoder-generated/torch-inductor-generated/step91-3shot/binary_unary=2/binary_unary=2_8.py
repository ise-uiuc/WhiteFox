
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 16, 5, stride=2, padding=2, dilation=2, groups=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1, dilation=1, groups=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 1, stride=2, padding=0, dilation=1, groups=1)
    def forward(self, x1):
        v1 = F.leaky_relu(self.conv1(x1), negative_slope=0.2)
        v2 = F.leaky_relu(self.conv2(v1), negative_slope=0.2)
        v3 = v2 - 0.5
        v4 = F.leaky_relu(v3, negative_slope=0.2)
        v5 = F.leaky_relu(self.conv3(v4), negative_slope=0.2)
        v6 = v5 - 0.5
        v7 = F.leaky_relu(v6, negative_slope=0.2)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
