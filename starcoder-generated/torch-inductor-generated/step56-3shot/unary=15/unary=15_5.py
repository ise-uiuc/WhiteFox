
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 8, 10, stride=5, padding=2)
        self.conv3 = torch.nn.Conv2d(8, 8, 10, stride=5, padding=2)
        self.conv4 = torch.nn.Conv2d(8, 1, 10, stride=5, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        v7 = self.conv4(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 227, 227)
