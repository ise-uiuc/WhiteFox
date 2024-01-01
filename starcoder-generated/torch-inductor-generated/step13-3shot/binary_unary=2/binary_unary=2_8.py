
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.BatchNorm2d(20)
        self.norm2 = torch.nn.BatchNorm2d(60)
        self.conv1 = torch.nn.Conv2d(3, 20, 5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(20, 60, 5, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.norm1(self.conv1(x1))
        v2 = v1 - 10
        v3 = F.relu(v2)
        v4 = self.norm2(self.conv2(v3))
        v5 = v4 - 11
        v6 = F.hardtanh(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
