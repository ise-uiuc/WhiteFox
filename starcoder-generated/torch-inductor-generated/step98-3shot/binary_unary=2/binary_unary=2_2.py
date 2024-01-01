
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(4, 4, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(4, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 0.05
        v4 = F.relu(v3)
        v5 = self.conv3(v4)
        v6 = v5 - 0.3
        v7 = F.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
