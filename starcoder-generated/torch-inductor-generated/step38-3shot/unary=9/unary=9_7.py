
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 9)
        self.conv3 = torch.nn.Conv2d(16, 32, 7)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = 3 + v1
        v3 = v2.clamp(min=0, max=6)
        v4 = v3 / 6
        v5 = self.conv2(v4)
        v6 = 3 + v5
        v7 = v6.clamp(min=0, max=6)
        v8 = v7 / 6
        v9 = self.conv3(v8)
        v10 = 3 + v9
        v11 = v10.clamp(min=0, max=6)
        v12 = v11 / 6
        return v12
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
