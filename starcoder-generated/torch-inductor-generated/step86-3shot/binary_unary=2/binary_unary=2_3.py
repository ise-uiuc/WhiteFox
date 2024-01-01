
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 256, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(256, 512, 1, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(512, 1024, 3, stride=0, padding=2)
        self.conv4 = torch.nn.Conv2d(1024, 2048, 1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(2048, 4096, 1, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 100
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 200
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 - 300
        v9 = F.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 - 400
        v12 = F.relu(v11)
        v13 = self.conv5(v12)
        v14 = v13 - 500
        v15 = F.relu(v14)
        return v15
# Inputs to the model
x1 = torch.randn(1, 10, 32, 32)
