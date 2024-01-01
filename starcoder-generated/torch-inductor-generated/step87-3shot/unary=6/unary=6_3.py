
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(3)
        self.conv3 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = 3 + v1
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.bn1(v6)
        v8 = torch.clamp_min(v7, 0)
        v9 = torch.clamp_max(v8, 6)
        v10 = v6 * v9
        v11 = v10 / 6
        v12 = self.conv2(v11)
        v13 = 3 + v12
        v14 = torch.clamp_min(v13, 0)
        v15 = torch.clamp_max(v14, 6)
        v16 = v12 * v15
        v17 = v16 / 6
        v18 = self.bn2(v17)
        v19 = torch.clamp_min(v18, 0)
        v20 = torch.clamp_max(v19, 6)
        v21 = v17 * v20
        v22 = v21 / 6
        v23 = self.conv3(v22)
        v24 = 3 + v23
        v25 = torch.clamp_min(v24, 0)
        v26 = torch.clamp_max(v25, 6)
        v27 = v23 * v26
        v28 = v27 / 6
        v29 = self.bn3(v28)
        v30 = torch.clamp_min(v29, 0)
        v31 = torch.clamp_max(v30, 6)
        v32 = v28* v31
        v33 = v32 / 6
        return v33
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
