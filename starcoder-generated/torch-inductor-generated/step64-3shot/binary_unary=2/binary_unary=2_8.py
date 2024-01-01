
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.25
        v3 = F.relu(v2)
        v4 = self.conv4(v3)
        v5 = v4 - 0.5
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 - 0.75
        v9 = F.relu(v8)
        v10 = self.conv2(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 128, 64, 64)
