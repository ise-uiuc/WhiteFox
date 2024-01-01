
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 620, 10, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(620, 1240, 10, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(1240, 1, 8, stride=2, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.14
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 4.0
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 10, 8, 8)
