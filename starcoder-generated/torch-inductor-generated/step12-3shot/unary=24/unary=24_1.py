
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x):
        negative_slope = 1
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v4 = v2 * 0.1
        v5 = torch.where(v2 > 0, v2, v4)
        return 0.5 + v5
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
