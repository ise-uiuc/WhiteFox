
class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 8, 9)
        self.conv1 = torch.nn.Conv2d(8, 8, 1)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = 7 + v1
        v3 = v2.clamp(min=0, max=6)
        v4 = v3 / 6
        v5 = self.conv1(v4)
        v6 = 7 + v5
        v7 = v6.clamp(min=0, max=6)
        v8 = v7 / 6
        return v8
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 8, 1)
        self.conv1 = torch.nn.Conv2d(8, 8, 9)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = 7 + v1
        v3 = v2.clamp(min=0, max=6)
        v4 = v3 / 6
        v5 = self.conv1(v4)
        v6 = 7 + v5
        v7 = v6.clamp(min=0, max=6)
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
