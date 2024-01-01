
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        v3 = v2 + x2
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = self.conv4(v5)
        v7 = v6 + x3
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
