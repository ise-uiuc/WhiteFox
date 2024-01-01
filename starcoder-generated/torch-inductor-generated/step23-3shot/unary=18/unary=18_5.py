
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn2 = torch.nn.BatchNorm2d(16)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = self.bn2(v2)
        v4 = self.conv2(v3)
        v5 = torch.sigmoid(v4)
        return v5
    def __init__(self):
        super().__init__()
        self.convt(self.conv2(self.conv1(v2)))
# Inputs to the model
x1 = torch.randn(1, 3, 1, 1)
