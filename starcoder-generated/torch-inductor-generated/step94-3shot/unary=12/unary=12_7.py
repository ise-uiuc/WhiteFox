
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 1, stride=1, padding=1, dilation=1, groups=2)
        self.conv2 = torch.nn.Conv2d(2, 5, 1, stride=1, padding=1, dilation=1, groups=2)
        self.conv3 = torch.nn.Conv2d(5, 3, 1, stride=1, padding=1, dilation=1, groups=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.sigmoid(v1)
        v3 = v1 * v2
        v33 = torch.cat([v3, v1], 1)
        v4 = self.conv2(v33)
        v5 = F.sigmoid(v4)
        v6 = v4 * v5
        v7 = torch.cat([v6, v1, v4, v3], 1)
        v8 = self.conv3(v7)
        v9 = F.sigmoid(v8)
        v10 = v8 * v9
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 2, 2)
