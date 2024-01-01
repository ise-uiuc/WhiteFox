
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 3, 1, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(x3)
        # v4 = x1 + x2
        v5 = v1 + v2 + v3
        v6 = self.conv4(v5)
        # v7 = x2 + x3
        v8 = v1 + v2
        v9 = v1 + v3
        v10 = v2 + v3
        v11 = v8 + v9 + v10
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 3, 16, 16)
x3 = torch.randn(1, 3, 16, 16)
