
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
    def forward(self, x1):
        x1 = torch.relu(self.bn1(self.conv1(x1)))
        x1 = torch.cat([x1, x1 + 5])
        x1 = x1 + torch.neg(x1)
        x1 = self.conv2(x1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
