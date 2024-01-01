
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 1, stride=2, padding=2)
        self.conv_next = torch.nn.Conv2d(64, 16, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = F.sigmoid(x1)
        v1 = v1.mul(x1)
        v2 = self.conv(v1)
        v3 = F.sigmoid(v2)
        v3 = v2.mul(v3)
        v4 = v2.add(v3)
        v5 = self.conv_next(v4)
        v6 = F.sigmoid(v5)
        v6 = v5.mul(v6)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
