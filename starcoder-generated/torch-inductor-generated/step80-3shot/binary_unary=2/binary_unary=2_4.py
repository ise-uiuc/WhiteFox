
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1)
        self.bn = torch.nn.BatchNorm2d(4)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.squeeze()
        v3 = v2.unsqueeze(1)
        v4 = torch.cat((v1, v3), dim=1)
        v5 = self.bn(v4)
        v6 = v5 - 1
        v7 = F.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
