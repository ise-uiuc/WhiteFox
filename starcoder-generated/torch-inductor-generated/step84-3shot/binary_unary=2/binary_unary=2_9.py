
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv1d(256, 256, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(256, 256, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv3d(256, 256, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = v1 - 0.73
        v3 = F.relu(v2)
        v4 = self.conv1(x1)
        v5 = self.conv2(v3)
        v6 = v5 - v4
        v7 = F.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 256, 128)
