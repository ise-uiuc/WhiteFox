
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(4, 4, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        a1 = self.conv2(x1)
        v4 = self.conv3(x1)
        v2 = v1 + a1
        v3 = torch.relu(v2)
        v5 = (v3 + v4) + x1
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
