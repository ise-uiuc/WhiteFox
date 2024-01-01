
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3)
        self.conv2 = torch.nn.Conv2d(16, 4, 1, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 10
        v3 = F.relu(v2)
        v4 = F.avg_pool2d(v3, 2, stride=2, padding=0)
        v5 = self.conv2(v4)
        v6 = v5 - 11
        v7 = F.relu(v6)
        v8 = v7 - 7
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
