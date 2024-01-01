
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, (9, 1), stride=1, padding=4)
        self.conv2 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(1, 1, (5, 1), stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv1(x1)
        v4 = v1 + v2 + v3
        v5 = torch.relu(v4)
        v6 = self.conv2(v4)
        v7 = v5 + v6
        v8 = self.conv3(v7)
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 96, 32)
