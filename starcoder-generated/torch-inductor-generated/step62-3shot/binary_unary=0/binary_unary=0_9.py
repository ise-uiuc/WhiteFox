
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv5 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.relu = torch.nn.ReLU()
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = self.conv2(v2)
        v4 = v2 + v3
        v5 = self.relu(v4)
        v6 = self.conv3(v5)
        v7 = v5 + v6
        v8 = self.relu(v7)
        v9 = self.conv4(v8)
        v10 = self.relu(v9)
        v11 = v10 + v3
        v12 = self.relu(v11)
        v13 = self.conv5(v12)
        v14 = torch.argmax(v13)
        v15 = self.conv3(v13)
        v16 = self.relu(v15)
        v17 = self.conv4(v16)
        v18 = v17 + torch.tensor(v14)
        v19 = torch.relu(v18)
        return v19
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
