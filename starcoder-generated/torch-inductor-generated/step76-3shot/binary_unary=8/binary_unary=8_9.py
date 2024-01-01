
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(4, 4, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(4, 4, 1, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(4, 4, 1, stride=1, padding=3)
        self.conv5 = torch.nn.Conv2d(4, 4, 1, stride=1, padding=5)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = self.conv4(x1)
        v5 = self.conv1(x1)
        v6 = v1 + v2 + v3 + v4 + v5
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 4, 100, 100)
