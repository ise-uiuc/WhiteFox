
class Model(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(16, 33, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(33, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool2(x)
        return x
# Inputs to the model
x = torch.randn(224, 224, 16)
