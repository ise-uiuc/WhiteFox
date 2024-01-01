
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(1, 10, 8)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(10)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
# Inputs to the model
torch.manual_seed(1)
x = torch.randn(1, 1, 3, 3)
