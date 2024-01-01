
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 1, stride=2, padding=2)
    def forward(self, x1):
        v1 = F.sigmoid(x1)
        v1 = v1.mul(x1)
        v2 = F.sigmoid(v1)
        v2 = v1.mul(v2)
        v3 = F.sigmoid(v2)
        v3 = v2.mul(v3)
        v4 = F.sigmoid(v3)
        v4 = v3.mul(v4)
        v5 = self.conv(v4)
        v6 = F.sigmoid(v5)
        v6 = v5.mul(v6)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
