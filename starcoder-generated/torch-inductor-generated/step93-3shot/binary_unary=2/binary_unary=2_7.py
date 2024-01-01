
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 50, 1, padding=0)
        self.conv2 = torch.nn.Conv2d(50, 160, 1, padding=0)
        self.conv3 = torch.nn.Conv2d(160, 96, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(96, 54, 1, padding=0)
        self.conv5 = torch.nn.Conv2d(54, 48, 1, padding=0)
        self.conv6 = torch.nn.Conv2d(48, 24, 3, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - -16.0
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - -50
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 - -2
        v9 = F.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 - -6
        v12 = F.leaky_relu(v11, 0.1)
        v13 = self.conv5(v12)
        v14 = v13 - 1.0
        v15 = F.leaky_relu(v14, 0.1)
        v16 = self.conv6(v15)
        v17 = v16 - 5.0
        v18 = F.leaky_relu(v17, 0.1)
        return v18
# Inputs to the model
x1 = torch.randn(1, 1, 56, 56)
