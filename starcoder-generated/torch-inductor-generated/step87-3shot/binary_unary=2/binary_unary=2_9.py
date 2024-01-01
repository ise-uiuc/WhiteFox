
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 2, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(1, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = self.conv2(v1)
        v4 = F.relu(v2 - v3)
        v5 = self.conv3(v2)
        v6 = F.relu(v5 - v1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
