
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(25, 54, 15, stride=2, padding=5)
        self.conv2 = torch.nn.Conv2d(54, 129, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(129, 257, 3, stride=2, padding=1)
    def forward(self, x):
        negative_slope = 8.496311
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 > 0
        v5 = v3 * negative_slope
        v6 = torch.where(v4, v3, v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 25, 37, 33)
