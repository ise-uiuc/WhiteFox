
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 2)
        self.bn = torch.nn.BatchNorm2d(8, eps=1e-5)
        self.conv2 = torch.nn.Conv2d(8, 8, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 8, 4, 4)
