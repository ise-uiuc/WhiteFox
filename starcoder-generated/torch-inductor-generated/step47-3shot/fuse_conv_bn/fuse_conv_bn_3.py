
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(1, 1, 2)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(1)
        torch.manual_seed(1)
        self.pool = torch.nn.MaxPool2d(2)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 64, 128)
