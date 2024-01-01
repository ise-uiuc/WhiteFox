
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(320, 1, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(128, 320, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 320, 2, 2)
