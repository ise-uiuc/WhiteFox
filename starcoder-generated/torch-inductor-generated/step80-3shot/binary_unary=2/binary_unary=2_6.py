
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = self.conv2(v2)
        v5 = F.relu(v3)
        v4 = F.relu(-(v5 - torch.ones_like(v5)))
        v6 = self.conv2(v4)
        v7 = F.relu(v6 - 10)
        v8 = F.relu(self.conv1(v7 - 11))
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
