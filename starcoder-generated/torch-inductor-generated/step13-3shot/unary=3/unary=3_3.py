
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(32, 512, 32, stride=2, padding=16)
        self.conv1 = torch.nn.Conv2d(512, 256, 1, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = x1 + x2
        v2 = self.conv0(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        v8 = self.conv1(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 32, 224, 224)
x2 = torch.randn(1, 32, 224, 224)
