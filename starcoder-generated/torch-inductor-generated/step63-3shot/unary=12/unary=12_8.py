
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1024, 256, 1, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()
        self.mul = torch.mul
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = self.mul(v1, v2)
        v4 = v3 * x2
        return v4
# Inputs to the model
x1 = torch.randn(1, 1024, 64, 64)
x2 = torch.randn(1, 1024, 64, 64)
