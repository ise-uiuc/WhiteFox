
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 9, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 9, 3, stride=2, padding=2, dilation=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 1.64
        v3 = F.leaky_relu(v2, negative_slope=0.1)
        v4 = self.conv2(x1)
        v5 = v4 - 2.2
        v6 = F.leaky_relu(v5, negative_slope=0.1)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
