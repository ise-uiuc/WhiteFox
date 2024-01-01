
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0, groups=1)
        self.conv2 = torch.nn.Conv2d(8, 1, 1, stride=1, padding=0, groups=1)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = x2 - 0.7
        x4 = F.relu(x3)
        x5 = self.conv2(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 3, 256, 64)
