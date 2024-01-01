
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 5, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(5, 3, 1, stride=2, padding=1)
    def forward(self, x):
        negative_slope = 0.1
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv2(v4)
        v6 = v5 > 0
        v7 = v5 * negative_slope
        v8 = torch.where(v6, v5, v7)
        v9 = self.conv3(v8)
        v10 = v9 > 0
        v11 = v5 * -0.1
        v12 = torch.where(v10, v9, v11)
        return v12
# Inputs to the model
x1 = torch.randn(2, 3, 220, 220)
