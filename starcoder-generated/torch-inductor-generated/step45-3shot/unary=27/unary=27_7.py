
class Model(torch.nn.Module):
    def __init__(self, min1, max1, min2, max2):
        super().__init__()
        self.t1_conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.t2_conv = torch.nn.Conv2d(2, 1, 1, stride=1, padding=0)
        self.t3_conv = torch.nn.Conv2d(2, 4, 2, stride=2, padding=0)
        self.t4_conv = torch.nn.Conv2d(1, 1, 3, stride=2, padding=0)
        self.min1 = min1
        self.max1 = max1
        self.min2 = min2
        self.max2 = max2
    def forward(self, x1):
        v1 = self.t1_conv(x1)
        v2 = torch.clamp_min(v1, self.min1)
        v3 = torch.clamp_max(v2, self.max1)
        v4 = self.t2_conv(v3)
        v5 = torch.clamp_min(v4, self.min2)
        v6 = torch.clamp_max(v5, self.max2)
        v7 = self.t3_conv(v6)
        v8 = self.t4_conv(v7)
        return v8
min1 = 0.8
max1 = 1
min2 = 0.1
max2 = 1
# Inputs to the model
x1 = torch.randn(1, 1, 353, 512)
