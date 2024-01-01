
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 64, 14, stride=1, padding=7)
        self.conv2 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = torch.reshape(v6, (1, 128, 13, 13))
        v8 = self.conv2(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
