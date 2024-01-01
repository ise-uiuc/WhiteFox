
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 3, stride=1, padding=0, dilation=2, groups=1, bias=0)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        f1 = torch.sub(v1, 0.5)
        v2 = self.sigmoid(f1)
        v3 = v1 * v2
        u1 = torch.sigmoid(v3)
        v4 = v2 * u1
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 128, 128)
