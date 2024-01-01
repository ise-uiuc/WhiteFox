
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(8, 8, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(8, 8, 5, padding=3)
        self.conv3 = torch.nn.Conv2d(8, 8, 5)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = v1.add(3)
        v3 = v2.clamp(min=0, max=6)
        v4 = v3 / 6
        v5 = self.conv1(v4)
        v6 = v5.add(3)
        v7 = v6.clamp(min=0, max=6)
        v8 = v7 / 6
        v9 = self.conv2(v8)
        v10 = v9.add(3)
        v11 = v10.clamp(min=0, max=6)
        v12 = v11 / 6
        v13 = self.conv3(v12)
        v14 = v13.add(3)
        v15 = v14.clamp(min=0, max=6)
        v16 = v15 / 6
        return v16
# Inputs to the model
x1 = torch.randn(12, 3, 64, 64)
