
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(128, 256, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = torch.relu(self.conv2(v1))
        v3 = torch.relu(self.conv3(v2))
        v4 = torch.relu(self.conv4(v3))
        v5 = torch.relu(self.conv5(v4))
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 56, 56)
