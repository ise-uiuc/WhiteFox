
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 20, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(50, 80, 3, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(80, 128, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 10
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 11
        v6 = F.relu(v5)
        # print(v6.shape)
        v7 = self.conv3(v6)
        v8 = v7 - 15
        v9 = F.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 - 20
        v12 = F.relu(v11)
        v13 = torch.chunk(v12, 3, dim=1)
        return v13[1]
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
