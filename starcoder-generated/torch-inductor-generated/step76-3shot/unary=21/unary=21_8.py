
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(32)
    def forward(self, x):
        x = self.bn(self.conv1(x))
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
