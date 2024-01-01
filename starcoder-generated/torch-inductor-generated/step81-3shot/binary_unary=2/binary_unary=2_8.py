
class Model(nn.Module):
    def __init__(self):# 83
        super().__init__()
        self.batchNorm2d = nn.BatchNorm2d(80)# 84
        self.conv2d = nn.Conv2d(4, 64, (4, 4), stride=(2, 2), bias=False)# 85
    def forward(self, x1):# 86
        x2 = self.batchNorm2d(x1)# 87
        # 89
        x3 = torch.add(x2, 1.0)# 90
        x4 = F.relu(x3)# 91
        v1 = self.conv2d(x4)# 92
        v2 = v1 - 0.25# 93
        v3 = F.relu(v2)# 94
        v4 = torch.squeeze(v3, 0)# 95
        return v4# 96

# Inputs to the model
x1 = torch.randn(1, 4, 128, 128)
