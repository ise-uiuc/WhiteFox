
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 20, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(20, 50, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(50, 100, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v1)
        v4 = torch.relu(v3)
        v5 = self.conv3(v3)
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 30, 30)
