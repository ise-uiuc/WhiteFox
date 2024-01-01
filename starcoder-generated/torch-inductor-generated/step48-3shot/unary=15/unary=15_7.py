
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv0(x)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = torch.relu(v7)
        v9 = self.conv4(v8)
        return v9
# Inputs to the model
x = torch.randn(1, 3, 310, 310)
