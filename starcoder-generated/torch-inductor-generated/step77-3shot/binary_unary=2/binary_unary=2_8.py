
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(6, 14, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(14, 14, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = F.relu(x1)
        v2 = self.conv1(v1)
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = F.relu(v4)
        v6 = self.conv3(v5)
        v7 = v6 - 10
        v8 = F.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 512, 512)
