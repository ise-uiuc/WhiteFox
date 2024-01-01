
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 6, 3, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(1, 16, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(6, 6, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(6, 6, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv2(x1)
        v4 = self.conv2(x1)
        v5 = self.conv3(torch.relu(v1))
        v6 = self.conv3(torch.relu(v2))
        v7 = self.conv4(torch.relu(v3) + torch.relu(v4) + v5 + v6 + torch.relu(v1))
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
