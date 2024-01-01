
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 4, stride=1)
        self.conv2 = torch.nn.Conv2d(256, 64, 4, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = torch.max(v6,) # maxpool2D(v6,,, stride=)
        v8 = self.conv2(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
