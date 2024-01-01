
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 1, 1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm1d(1)
        self.conv2 = torch.nn.Conv1d(1, 4, 1, stride=1, padding=0)
        self.bn2 = torch.nn.BatchNorm1d(4)
    def forward(self, x):
        negative_slope = 0.01164564
        v1 = self.conv1(x)
        v2 = self.bn1(v1)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        v6 = self.conv2(v5)
        v7 = self.bn2(v6)
        return v7
# Inputs to the model
x1 = torch.randn(5, 1, 800)
