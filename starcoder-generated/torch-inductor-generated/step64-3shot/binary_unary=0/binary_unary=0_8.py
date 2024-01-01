
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(x2)
        v4 = torch.relu(v3)
        v5 = self.conv2(x3)
        v6 = torch.relu(v5)
        v7 = self.conv2(x4)
        v8 = torch.relu(v7)
        v9 = v2 + v8
        v10 = torch.relu(v9)
        v12 = self.conv3(v10)
        v13 = torch.relu(v12)
        v15 = self.conv3(v13)
        v16 = torch.relu(v15)
        v18 = self.conv3(v16)
        v19 = torch.relu(v18)
        v20 = v19 + v4
        v21 = torch.relu(v20)
        v23 = self.conv4(v21)
        v24 = torch.nn.ReLU()(v23)
        v25 = v24 * x1
        v26 = self.conv3(v25)
        v27 = torch.relu(v26)
        v29 = self.conv3(v27)
        v30 = torch.relu(v29)
        v32 = self.conv3(v30)
        v33 = torch.relu(v32)
        v34 = self.conv4(v33)
        v35 = self.conv1(v34)
        return v35
# Input to the model
x1 = torch.randn(1, 16, 64, 64)
