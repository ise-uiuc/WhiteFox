
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(1, 20, 5, 1)
        self.bn2d = nn.BatchNorm2d(20)

    def forward(self, x):
        y = self.conv2d(x)
        z = self.bn2d(y)
        return z
# Inputs to the model
x = torch.randn(1, 1, 28, 28)
