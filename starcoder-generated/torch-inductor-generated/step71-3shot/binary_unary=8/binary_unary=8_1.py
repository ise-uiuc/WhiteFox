
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv1(x1)
        v4 = self.conv1(x1)
        v5 = self.conv1(x1)
        v6 = v1 + v2 + v3
        v7 = torch.relu(v6)
        v8 = self.conv1(x1)
        v9 = self.conv1(x1)
        v10 = self.conv1(x1)
        v11 = self.conv1(x1)
        v12 = self.conv1(x1)
        v13 = v8 + v9 + v10
        v14 = torch.relu(v13)
        v15 = self.conv1(x1)
        v16 = self.conv1(x1)
        v17 = self.conv1(x1)
        v18 = self.conv1(x1)
        v19 = self.conv1(x1)
        v20 = v15 + v16 + v17
        v21 = torch.relu(v20)
        v22 = self.conv1(x1)
        v23 = self.conv1(x1)
        v24 = self.conv1(x1)
        v25 = self.conv1(x1)
        v26 = self.conv1(x1)
        v27 = v22 + v23 + v24
        v28 = torch.relu(v27)
        v29 = v7 + v14 + v21 + v28
        v30 = torch.relu(v29)
        v31 = v30.flatten(1)
        v32 = v31.add_(3.14e+00)
        v33 = F.relu(v32)
        return v33
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
