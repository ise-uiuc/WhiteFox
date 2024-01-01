
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(37)
        self.conv1 = torch.nn.Conv2d(37, 112, 2, stride=2, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(112)
        self.conv2 = torch.nn.Conv2d(112, 288, 2, stride=2, padding=0)
        self.bn3 = torch.nn.BatchNorm2d(288)
        self.conv3 = torch.nn.Conv2d(288, 624, 2, stride=2, padding=0)
        self.bn4 = torch.nn.BatchNorm2d(624)
        self.conv4 = torch.nn.Conv2d(624, 1296, 2, stride=2, padding=0)
        self.bn5 = torch.nn.BatchNorm2d(1296)
        self.conv5 = torch.nn.Conv2d(1296, 2332, 2, stride=2, padding=0)
    def forward(self, x162):
        v1 = self.bn1(x162)
        v2 = self.conv1(v1)
        v4 = self.bn2(v2)
        v5 = self.conv2(v4)
        v7 = self.bn3(v5)
        v8 = self.conv3(v7)
        v10 = self.bn4(v8)
        v11 = self.conv4(v10)
        v13 = self.bn5(v11)
        v14 = self.conv5(v13)
        v16 = v14 * 0.5
        v17 = v14 * v14
        v18 = v17 * v14
        v19 = v18 * 0.044715
        v20 = v14 + v19
        v21 = v20 * 0.7978845608028654
        v22 = torch.tanh(v21)
        v23 = v22 + 1
        v24 = v16 * v23
        return v24
# Inputs to the model
x162 = torch.randn(1, 37, 38, 62)
