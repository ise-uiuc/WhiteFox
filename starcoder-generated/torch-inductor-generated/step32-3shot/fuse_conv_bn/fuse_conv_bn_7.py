
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 2)
        self.conv2 = torch.nn.Conv2d(1, 1, 2)
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x):
        x1 = self.conv1(x)
        x1, _ = torch.min(x1, dim=3, keepdim=True)
        x2 = self.conv2(x1)
        x2, _ = torch.max(x2, dim=2, keepdim=True)
        y = self.bn(x2)
        return y
# Inputs to the model
x1 = torch.randn(1, 1, 5, 3)
