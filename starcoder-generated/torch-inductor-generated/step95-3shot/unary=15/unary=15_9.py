
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(64, 96, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(96, 128, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = torch.relu(self.conv2(v1))
        v3 = torch.relu(self.conv3(v2))
        v4 = torch.relu(self.conv4(v3))
        v5 = torch.relu(self.conv5(v4))
        v6 = torch.relu(self.conv6(v5))
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
