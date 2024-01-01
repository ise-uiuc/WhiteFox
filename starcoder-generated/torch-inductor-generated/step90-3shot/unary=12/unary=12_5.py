
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 24, 1, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.05
        v3 = self.sigmoid(v2)
        v4 = torch.mul(v2, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 176, 176)
