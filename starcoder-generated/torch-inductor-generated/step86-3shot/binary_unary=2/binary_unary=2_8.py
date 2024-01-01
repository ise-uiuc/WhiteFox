
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 7, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(128, 256, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1) - 500
        v2 = F.relu(v1)
        v3 = self.conv2(v2 - 150) - 120
        v4 = F.relu(v3)
        v5 = self.conv3(v4) - 300
        return v5
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
