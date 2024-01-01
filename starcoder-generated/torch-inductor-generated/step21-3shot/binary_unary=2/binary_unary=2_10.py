
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=2, groups=2)
        self.conv2 = torch.nn.Conv2d(32, 32, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.75
        v3 = F.relu(v2)
        return self.conv2(v3)
# Inputs to the model
x1 = torch.randn(1, 32, 4, 4)
