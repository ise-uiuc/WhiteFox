
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.conv2 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.bn2 = torch.nn.BatchNorm2d(4)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = self.conv2(v2)
        v4 = self.bn2(v3)
        v5 = F.dropout(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
