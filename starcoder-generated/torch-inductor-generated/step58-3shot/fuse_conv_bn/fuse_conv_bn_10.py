
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x2):
        x2 = self.conv(x2)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x2 = self.conv(x2)
        x2 = self.bn(x2)
        return x2
# Inputs to the model
x2 = torch.randn(1, 3, 3, 3)
