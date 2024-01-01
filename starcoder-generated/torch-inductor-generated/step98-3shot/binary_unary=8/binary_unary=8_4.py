
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 32, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(256, 64, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(1, 168, 1, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv1(x1)
        v4 = self.conv2(v3)
        v5 = self.conv1(x2)
        v6 = self.conv2(v5)
        v7 = self.conv1(x2)
        v8 = self.conv3(v4)
        v9 = torch.cat((v2, v8))
        v10 = self.conv4(v9)
        v11 = torch.relu(v10)
        v12 = self.conv2(v11)
        v13 = self.conv4(v12)
        v14 = torch.relu(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 256, 256)
