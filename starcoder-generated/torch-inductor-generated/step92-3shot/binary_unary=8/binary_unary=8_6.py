
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 11, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 11, 3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(3, 11, 3, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(3, 11, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = self.conv4(x1)
        v5 = self.conv1(x1)
        v6 = self.conv2(x1)
        v7 = self.conv3(x1)
        v8 = self.conv4(x1)
        v9 = self.conv1(x1)
        v10 = self.conv2(x1)
        v11 = self.conv3(x1)
        v12 = self.conv4(x1)
        v13 = v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10 + v11 + v12
        v14 = torch.relu(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
