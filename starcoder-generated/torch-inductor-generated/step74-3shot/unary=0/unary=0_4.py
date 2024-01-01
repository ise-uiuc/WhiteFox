
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(255, 171, 16, stride=1, padding=2)
        self.conv2d = torch.nn.Conv2d(27, 61, 63, stride=24, padding=14)
        self.conv3d = torch.nn.Conv3d(108, 28, 78, stride=88, padding=107)
    def forward(self, x920):
        v1 = self.conv1d(x920)
        v2 = self.conv2d(x920)
        v3 = self.conv3d(x920)
        v4 = v1 * v2
        v5 = v2 * v1
        v6 = v1 - v5
        v7 = v6 * v3
        return v7
# Inputs to the model
x920 = torch.randn(1, 255, 308)
