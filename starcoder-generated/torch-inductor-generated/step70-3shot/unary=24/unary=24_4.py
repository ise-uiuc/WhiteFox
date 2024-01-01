
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 10, stride=5, padding=5)
        self.conv2 = torch.nn.Conv2d(16, 28, 5, stride=4, padding=4)
        self.conv3 = torch.nn.Conv2d(28, 28, 3, stride=3, padding=1)
        self.conv4 = torch.nn.Conv2d(28, 6, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(6, 8, 1, stride=1, padding=0)
    def forward(self, x):
        negative_slope = 0.27705123
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv2(v4)
        v6 = v5 > 0
        v7 = v5 * 0.52104077
        v8 = torch.where(v6, v5, v7)
        v9 = self.conv3(v8)
        v10 = v9 > 0
        v11 = v9 * 0.7293782
        v12 = torch.where(v10, v9, v11)
        v13 = self.conv4(v12)
        v14 = v13 > 0
        v15 = v13 * 0.22378
        v16 = torch.where(v14, v13, v15)
        v17 = self.conv5(v16)
        v18 = v17 > 0
        v19 = v17 * 0.849492
        v20 = torch.where(v18, v17, v19)
        return v20
# Inputs to the model
x1 = torch.randn(20, 3, 27, 61)
