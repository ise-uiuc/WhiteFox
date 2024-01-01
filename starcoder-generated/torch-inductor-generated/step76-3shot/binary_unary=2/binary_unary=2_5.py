
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 25, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = v2 - 34
        v4 = self.conv1(v2)
        v5 = F.relu(v4)
        v6 = F.avg_pool2d(x1, 3, stride=2)
        v7 = v6 - 20
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 40, 40)
