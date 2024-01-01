
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, (3, 5), stride=1, padding=(1, 2))
        self.conv3 = torch.nn.Conv2d(3, 3, (1, 3), stride=1, padding=(0, 1))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 7.5
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 15
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 - 22.5
        v9 = F.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
