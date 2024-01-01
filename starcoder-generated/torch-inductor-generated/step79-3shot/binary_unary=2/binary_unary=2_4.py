
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 4, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(64, 64, 6, stride=1, padding=2)
        self.conv5 = torch.nn.Conv2d(64, 64, 7, stride=2, padding=4)
        self.conv6 = torch.nn.Conv2d(64, 64, 9, stride=3, padding=5)
        self.conv7 = torch.nn.Conv2d(64, 64, 16, stride=3, padding=5)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.7
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 0.7
        v6 = F.relu(v5)
        v7 = self.conv3(x1 * 2)
        v8 = x1 - 0.7
        v9 = F.relu(v8)
        v10 = self.conv4(v9 + x1)
        v11 = v10 - x1
        v12 = F.relu(v4)
        v13 = v12 + 0.1
        v14 = torch.clamp(v13, min=-2.0)
        v15 = x1 - 0.7
        v16 = F.relu(v15)
        v17 = self.conv5(x1)
        v18 = v17 - 0.7
        v19 = F.relu(v18)
        v20 = self.conv6(x1)
        v21 = v20 - 0.7
        v22 = F.relu(v21)
        v23 = self.conv7(x1)
        v24 = v23 - 0.7
        v25 = F.relu(v3)
        v26 = x1 - 0.7
        v27 = F.relu(v10)
        v28 = torch.clamp(v25, min=0.1)
        v29 = self.conv1(x1 * 2)
        v30 = F.relu(v29)
        v31 = x1 - torch.tensor(2.0)
        v32 = F.relu(v27)
        v33 = x1 - torch.tensor(2.0)
        v34 = torch.clamp(v25, max=0.1)
        v35 = self.conv1(x1 + v30)
        v36 = F.relu(v35)
        v37 = self.conv2(v36)
        v38 = v37 + 10
        v39 = F.relu(v19)
        v40 = self.conv3(v39)
        v41 = v6 - 0.7
        v42 = F.relu(v40)
        v43 = self.conv4(v42)
        v44 = v40 - torch.tensor(2.0)
        v45 = F.relu(v24)
        return v43
# Inputs to the model
x1 = torch.randn(1, 3, 48, 48)
