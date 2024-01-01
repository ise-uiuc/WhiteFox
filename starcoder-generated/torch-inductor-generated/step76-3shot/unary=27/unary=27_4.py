
class Model(torch.nn.Module):
    def __init__(self, min1, max1, min2, max2, min3, max3):
        super().__init__()
        self.conv2d_1 = torch.nn.Conv2d(7, 3, 1, 1, 0, 1)
        self.conv2d_2 = torch.nn.Conv2d(16, 2, 2, 1, 0, 1)
        self.conv2d_3 = torch.nn.Conv2d(19, 4, 4, 1, 0, 0)
        self.min1 = min1
        self.max1 = max1
        self.min2 = min2
        self.max2 = max2
        self.min3 = min3
        self.max3 = max3
def forward(self, x0):
        v0 = self.conv2d_1(x0)
        v4 = torch.clamp_max(v0, self.max1)
        v5 = torch.clamp_min(v4, self.min1)
        v1 = self.conv2d_2(v5)
        v6 = torch.clamp_max(v1, self.max2)
        v7 = torch.clamp_min(v6, self.min2)
        v2 = self.conv2d_3(v7)
        v8 = torch.clamp_max(v2, self.max3)
        v9 = torch.clamp_min(v8, self.min3)
        return v9
        x0 = torch.randn(1, 7, 8, 6)
min1 = 0.65
max1 = -0.892625
min2 = 0.3
max2 = 0.765
min3 = 0.593
max3 = 0.963
# Inputs to the model
