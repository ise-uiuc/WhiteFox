
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 1, 7, stride=1, padding=3)
        self.relu = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(1, 1, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.relu(x1)
        v2 = self.conv1(v1)
        v3 = self.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.relu(v4)
        v6 = self.conv1(v5)
        v7 = self.relu(v1)
        v8 = self.conv3(v7)
        v9 = v8 * v6
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
