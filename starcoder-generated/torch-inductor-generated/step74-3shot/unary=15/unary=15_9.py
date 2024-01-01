
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.maxpool1 = torch.nn.MaxPool2d(3)
        self.conv2 = torch.nn.Conv2d(16, 64, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.maxpool2 = torch.nn.MaxPool2d(3)
        self.gpool = gpool_p2
        self.conv4 = torch.nn.Conv2d(128, 256, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(256, 512, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.maxpool1(v2)
        v4 = self.conv2(v3)
        v5 = torch.relu(v4)
        v6 = self.conv3(v5)
        v7 = torch.relu(v6)
        v8 = self.maxpool2(v7)
        v9 = self.gpool(v8)
        v10 = self.conv4(v9)
        v11 = torch.relu(v10)
        v12 = self.conv5(v11)
        v14 = torch.relu(v12)
        return v14
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
