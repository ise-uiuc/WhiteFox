
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, stride=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 3, stride=1)
        self.conv3 = torch.nn.Conv2d(4, 6, 3, stride=1)
        self.conv4 = torch.nn.Conv2d(6, 6, 3, stride=1)
        self.conv5 = torch.nn.Conv2d(6, 10, 3, stride=1)
        self.conv6 = torch.nn.Conv2d(10, 10, 3, stride=1)
        self.conv7 = torch.nn.Conv2d(10, 48, 7, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = self.conv2(v2)
        v4 = F.relu(v3)
        v5 = self.conv3(v4)
        v6 = F.relu(v5)
        v7 = self.conv4(v6)
        v8 = F.relu(v7)
        v9 = self.conv5(v8)
        v10 = F.relu(v9)
        v11 = self.conv6(v10)
        v12 = F.relu(v11)
        v13 = self.conv7(v12)
        v14 = F.relu(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
