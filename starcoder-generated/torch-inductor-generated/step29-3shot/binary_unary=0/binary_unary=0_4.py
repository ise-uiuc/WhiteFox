
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.relu(v1)
        v3 = F.avg_pool2d(v1, 3, stride=1, padding=1)
        v4 = F.conv2d(v3, torch.randn(16, 16, 3, 3), stride=1, padding=1)
        v5 = F.relu(v4)
        v6 = F.adaptive_avg_pool2d(v5, (7, 7))
        v7 = F.sigmoid(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 16, 28, 28)
