
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 512, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 512, 3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 512, 3, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(64, 512, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv1(x1)
        v4 = self.conv1(x1)
        v5 = self.conv1(x3)
        v6 = v1 + v2 + v3 + v4
        v7 = v6 + v5
        v8 = torch.relu(v7)
        v9 = self.conv2(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 64, 20, 20)
x2 = torch.randn(1, 8, 16, 16)
x3 = torch.randn(1, 64, 64, 64)
