
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x):
        v = self.conv1(x)
        v1 = v * 2
        v2 = v + x
        v3 = x + v2
        v4 = self.conv2(v1)
        v5 = self.conv3(v4)
        v6 = v5 + v
        v7 = self.conv4(v3)
        v8 = self.conv5(v7)
        v9 = self.conv6(v8)
        v10 = v9 + v5
        v11 = self.conv7(v10)
        v12 = v11 + v10
        return v12
# Inputs to the model
x = torch.randn(1, 16, 32, 32)
