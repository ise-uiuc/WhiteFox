
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = torch.clamp(v1, min=0, max=3)
        v4 = torch.mul(v3, v2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
