
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(22, 5, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 22, 1, stride=1, padding=0)
    def forward(self, x):
        negative_slope = 0.3064838
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = self.conv2(torch.where(v2, v1, v3))
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 2, 4)
