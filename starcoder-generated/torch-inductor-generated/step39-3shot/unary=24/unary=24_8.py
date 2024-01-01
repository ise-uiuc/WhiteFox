
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(256, 64, 1, stride=1, padding=0)
    def forward(self, x):
        negative_slope = 1.011716
        v1 = self.conv(x)
        v7 = self.conv(x)
        v8 = torch.sub(1, v7)
        v2 = v1.type(torch.LongTensor) > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = v4 > v8
        v9 = v4 * v8
        v6 = torch.where(v5, v4, v9)
        return v6
# Inputs to the model
x1 = torch.randn(1, 256, 7, 7)
