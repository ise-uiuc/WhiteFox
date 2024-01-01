
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
        self.conv3 = torch.nn.MaxPool2d(3, stride=1, padding=0)
        self.conv4 = torch.nn.MaxPool2d(3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(v1)
        v4 = self.conv4(v2)
        v5 = self.conv1(x1)
        v6 = self.conv2(x1)
        v7 = self.conv3(v5)
        v8 = self.conv4(v6)
        v9 = self.conv1(x1)
        v10 = self.conv2(x1)
        v11 = self.conv3(v9)
        v12 = self.conv4(v10)
        v13 = self.conv3(v4)
        v14 = self.conv4(v8)
        v15 = v3 + v4
        v16 = torch.relu(v15)
        v17 = self.conv3(v4)
        v18 = self.conv4(v8)
        v19 = self.conv3(v4)
        v20 = self.conv4(v8)
        v21 = self.conv3(v4)
        v22 = self.conv4(v8)
        v23 = self.conv3(v4)
        v24 = self.conv4(v8)
        v25 = v11 + v12 + v13 + v14 + v16 + v17 + v18 + v19 + v20 + v21 + v22 + v23 + v24
        v26 = torch.relu(v25)
        return v26
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
