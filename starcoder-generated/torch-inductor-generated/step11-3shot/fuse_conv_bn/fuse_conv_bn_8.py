
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.BatchNorm2d(1, momentum=0.5)
        self.conv = torch.nn.Conv2d(1, 1, 7)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.a(x1)
        x2 = self.relu(x1)
        x2 = self.bn(x2)
        return x2, x2
# Inputs to the model
x = torch.randn(3, 1, 10, 20)
