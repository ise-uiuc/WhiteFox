
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 18, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, 18, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 200
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 300
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 32, 56, 56)
