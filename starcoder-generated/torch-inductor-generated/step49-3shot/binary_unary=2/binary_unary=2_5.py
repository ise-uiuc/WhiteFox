
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 7, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 10
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 9
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 - 721
        v9 = F.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 - 13
        v12 = F.relu(v11)
        v13 = torch.squeeze(v12, 0)
        return v13
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
