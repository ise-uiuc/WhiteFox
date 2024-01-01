
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = self.conv_2(v1)
        v4 = v1 > 0
        v5 = v3 * 0.1
        v6 = torch.where(v5, v1, v3)
        v7 = torch.where(v4, v3, v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
