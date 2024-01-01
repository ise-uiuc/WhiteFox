
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        x = x.detach()
        x = self.conv1(x)
        x = self.bn1(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
