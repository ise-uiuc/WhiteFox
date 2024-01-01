
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.bn2 = torch.nn.BatchNorm2d(3)
        self.relu2 = torch.nn.ReLU(inplace=False)
        self.bn3 = torch.nn.BatchNorm2d(3)
        self.relu3 = torch.nn.ReLU(inplace=False)
        self.bn4 = torch.nn.BatchNorm2d(3)
        self.relu4 = torch.nn.ReLU(inplace=False)
        self.bn5 = torch.nn.BatchNorm2d(3)
        self.relu5 = torch.nn.ReLU(inplace=False)
        self.bn6 = torch.nn.BatchNorm2d(3)
        self.relu6 = torch.nn.ReLU(inplace=False)
    def forward(self, x1):
        # This model contains multiple points of insertion and some of them apply the relu op
        v1 = self.conv(x1)
        v2 = 3 + v1
        #v3 = torch.clamp_min(v2, 0)
        #v4 = torch.clamp_max(v3, 6)
        #v5 = v1 * v4
        v6 = v2 / 6
        v7 = self.bn1(v6)
        v8 = self.relu1(v7)
        v9 = self.bn2(v8)
        v10 = self.relu2(v9)
        v11 = v10 / 6
        v12 = self.bn3(v11)
        v13 = self.relu3(v12)
        v14 = self.bn4(v13)
        v15 = self.relu4(v14)
        v16 = v15 / 6
        v17 = self.bn5(v16)
        t1 = self.relu5(v17)
        v18 = torch.clamp_min(t1, 0)
        v19 = torch.clamp_max(v18, 6)
        v20 = v16 * v19
        v21 = v20 / 6
        v22 = self.bn6(v21)
        v23 = self.relu6(v22)
        return v23
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
