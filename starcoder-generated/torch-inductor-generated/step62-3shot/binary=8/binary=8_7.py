
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 2, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(2, 8, 3, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 + v1
        v5 = self.conv3(x2)
        v6 = self.conv2(v4)
        v7 = v2 + v5
        v8 = self.conv3(v7)
        v9 = self.conv1(v6)
        v10 = v6 + v8
        v11 = self.conv3(v3)
        v12 = self.conv1(v11)
        v13 = v8 + v10
        return v13
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
