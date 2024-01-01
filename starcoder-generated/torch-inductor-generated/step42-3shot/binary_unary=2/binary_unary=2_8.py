
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 50, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 10.5
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = F.relu(v4)
        v6 = -10.01 * torch.ones_like(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
