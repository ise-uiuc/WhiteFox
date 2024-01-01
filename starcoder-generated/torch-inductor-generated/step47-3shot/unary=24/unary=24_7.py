
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 1, (3, 3), stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 2, (2, 2), stride=(2, 2), padding=2, dilation=2)
    def forward(self, x):
        negative_slope = -1.9890641
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv2(v4)
        v6 = v5 > 0
        v7 = v5 * negative_slope
        v8 = torch.where(v6, v5, v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 2, 7, 7)
