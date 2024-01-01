
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(18, 27, 2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(27, 12, 4, stride=4, padding=0)
        self.conv3 = torch.nn.Conv2d(12, 3, 4, stride=4, padding=0)
    def forward(self, x):
        negative_slope = 0.8025886
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 > 0
        v5 = v3 * negative_slope
        v6 = torch.where(v4, v3, v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 18, 15, 6)
