
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 2, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 5, stride=5, padding=5)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.7
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 0.5
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
