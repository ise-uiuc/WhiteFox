
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 100, 3, stride=12, padding=5)
        self.conv2 = torch.nn.Conv2d(100, 100, 3, stride=12, padding=10)
        self.conv3 = torch.nn.Conv2d(100, 100, 3, stride=3, padding=2)
        self.conv4 = torch.nn.Conv2d(100, 200, 3, stride=7, padding=5)
        self.conv5 = torch.nn.Conv2d(200, 50, 3, stride=3, padding=6)
        self.conv6 = torch.nn.Conv2d(50, 2, 3, stride=6, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - torch.tensor(-0.36)
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - torch.tensor(-0.18)
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = self.conv3(v7)
        v9 = v7 + v8
        v10 = F.relu(v9)
        v11 = self.conv4(v10)
        v12 = self.conv4(v11)
        v13 = self.conv4(v12)
        v14 = v11 + v13
        v15 = F.relu(v14)
        v16 = self.conv5(v15)
        v17 = v1 + v16 + v16
        v18 = F.relu(v17)
        v19 = self.conv6(v18)
        v20 = F.log_softmax(v19, dim=1)
        return v20
# Inputs to the model
x1 = torch.randn(1, 1, 512, 512)
