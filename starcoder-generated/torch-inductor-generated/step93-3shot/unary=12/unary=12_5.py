
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1.sigmoid()
        v3 = v2 * v1
        v4 = self.conv2(v3)
        v5 = v4.sigmoid()
        v6 = v5 * v4
        v7 = self.conv3(v6)
        v8 = v7.sigmoid()
        v9 = v7 * v8
        v10 = self.conv4(v9)
        v11 = v10.sigmoid()
        v12 = v10 * v11
        return v12
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
