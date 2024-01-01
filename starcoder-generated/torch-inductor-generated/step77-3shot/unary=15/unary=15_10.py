
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2a = torch.nn.Conv2d(32, 64, 1, stride=2, padding=0)
        self.conv2b = torch.nn.Conv2d(32, 64, 1, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(256, 128, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2a = self.conv2a(v1)
        v2b = self.conv2b(v1)
        v3 = torch.cat([v2a, v2b], 1)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
