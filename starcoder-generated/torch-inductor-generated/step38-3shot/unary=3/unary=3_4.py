
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 2, stride=4, padding=8)
        self.conv2 = torch.nn.Conv2d(3, 8, 14, stride=15, padding=1)
        self.conv3 = torch.nn.Conv2dTranspose2d(8, 5, 1, stride=7, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.5011511847905441
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        v13 = self.conv3(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 1, 188, 192)
