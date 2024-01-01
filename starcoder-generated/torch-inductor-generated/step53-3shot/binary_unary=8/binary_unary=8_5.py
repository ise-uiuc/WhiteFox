
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 1, 3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 1, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv1(x1)
        v4 = self.conv2(x1)
        v5 = v4 + v4 + v4 + v4 + v4
        v6 = torch.relu(v5)
        return v1 + v2 + v3
# Inputs to the model
x1 = torch.randn(1, 16, 224, 224)
