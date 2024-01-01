
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 8, 5, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 1.4
        v3 = F.relu(v2)
        v4 = v1 - 1.0
        v5 = F.relu(v4)
        v7 = v3 + v5
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
