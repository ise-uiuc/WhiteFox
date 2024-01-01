
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, (1, 1))
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.conv2 = torch.nn.Conv2d(1, 1, (1, 1))
        self.bn2 = torch.nn.BatchNorm2d(1)
    def forward(self, x1, x2, inp=None):
        if inp is not None:
            v4 = (self.bn2(self.conv2(x1)))
        else:
            v4 = self.bn2(self.conv2(x1))
        v1 = self.conv1(torch.relu(self.bn1(v4)))
        v1 = v1 + inp
        return v1
# Inputs to the model
x1 = torch.randn(4, 4, 1, 1)
x2 = torch.randn(4, 3, 1, 1)
inp1 = torch.randn(1, 4, 2, 2)
inp2 = torch.randn(3, 4, 2, 2)
