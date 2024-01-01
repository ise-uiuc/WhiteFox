
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = 3 + v2
        v4 = v3.clamp(min=0, max=6)
        v5 = v4 / 6
        v6 = self.conv3(v5)
        v7 = 3 + v6
        v8 = v7.clamp(min=0, max=6)
        v9 = v8 / 6
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
