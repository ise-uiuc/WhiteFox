
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 256, 7, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 64, 7, stride=2, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 1
        v3 = F.relu(v2)
        v4 = v3.repeat(2, 1, 1, 1)
        v5 = self.conv2(v4)
        v6 = v5 - 5
        v7 = F.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 64, 224, 224)
