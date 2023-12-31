
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv1(x2)
        v3 = self.conv1(v2)
        v4 = self.conv1(v1)
        v5 = self.conv1(v3)
        v6 = self.conv1(v3)
        v7 = self.conv1(v4)
        v8 = self.conv1(v3)
        v9 = self.conv1(v7)
        v10 = self.conv1(v6)
        v11 = self.conv1(v2)
        v12 = self.conv1(v3)
        v13 = self.conv1(v8)
        v14 = self.conv1(v3)
        v15 = self.conv1(v3)
        v16 = self.conv1(v12)
        v17 = self.conv1(v10)
        v18 = self.conv1(v5)
        v19 = self.conv1(v14)
        v20 = self.conv1(v16)
        v21 = self.conv1(v1)
        v22 = self.conv1(v9)
        v23 = self.bn2(torch.add(v11, v13))
        v24 = self.bn1(torch.add(torch.mul(v17, v17), v23))
        v25 = self.bn1(v20)
        v26 = self.bn1(v15)
        v27 = self.bn1(v24)
        v28 = self.bn1(torch.mul(v19, v19))
        v29 = self.bn1(v25)
        v30 = self.bn2(v18)
        v31 = self.bn1(v29)
        v32 = self.bn2(v31)
        v33 = self.bn2(v27)
        v34 = self.bn1(torch.mul(torch.add(v18, v18), torch.add(v26, v26)))
        return torch.add(self.bn2(v30), torch.mul(v32, v34))
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
