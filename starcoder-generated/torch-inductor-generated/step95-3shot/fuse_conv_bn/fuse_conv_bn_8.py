
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, (3,5), 1, padding=(1,2), dilation=(2,1), groups=4)
        self.bn1 = torch.nn.BatchNorm2d(32)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return x
# Inputs to the model
x = torch.randn(5, 16, 55, 20)
