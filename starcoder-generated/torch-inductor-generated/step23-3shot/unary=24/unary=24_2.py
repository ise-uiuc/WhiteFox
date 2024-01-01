
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x, negative_slope):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 * negative_slope
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x = 1
