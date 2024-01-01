
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 3, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return 1 + v4
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
