
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 7, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 100
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 200
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
