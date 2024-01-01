
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(13, 8, 1, stride=1, padding=1)
    def forward(self, x1, x1_1, x2, x2_1, x3, x3_1, x4, x4_1):
        v1 = self.conv(x1)
        v2 = self.conv(x2)
        v3 = self.conv(x3)
        v4 = self.conv(x4)
        if x1_1 == None:
            x1_1 = torch.randn(v1.shape)
        v5 = v1 + x1_1
        if x2_1 == None:
            x2_1 = torch.randn(v2.shape)
        v6 = v2 + x2_1
        if x3_1 == None:
            x3_1 = torch.randn(v3.shape)
        v7 = v3 + x3_1
        if x4_1 == None:
            x4_1 = torch.randn(v4.shape)
        v8 = v4 + x4_1
        return torch.cat([v5, v6, v7, v8], dim=1)
# Inputs to the model
x1 = torch.randn(1, 13, 64, 64)
x1_1 = torch.randn(1, 13, 64, 64)
x2 = torch.randn(1, 13, 64, 64)
x2_1 = torch.randn(1, 13, 64, 64)
x3 = torch.randn(1, 13, 64, 64)
x3_1 = torch.randn(1, 13, 64, 64)
x4 = torch.randn(1, 13, 64, 64)
x4_1 = torch.randn(1, 13, 64, 64)
