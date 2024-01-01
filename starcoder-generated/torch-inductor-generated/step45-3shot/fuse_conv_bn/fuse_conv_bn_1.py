
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(12, 12, 1)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(12)
    def forward(self, x):
        x = self.conv(self.conv(x))
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 12, 32, 32)
