
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_relu = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU9x4())
        self.pooling = nn.Sequential(nn.MaxPool2d(10, 2, 15, 0), nn.MaxPool2d(4, 4, 3, 3), nn.MaxPool2d(7, 2, 0, 1), nn.MaxPool2d(3, 1, 1, 1), nn.MaxPool2d(5, 4, 2, 2))
        self.concat = nn.Sequential(nn.Conv2d(136, 63, 3, 1, 1), nn.Conv2d(63, 103, 3, 1, 1))
    def forward(self, x):
        x = self.conv_relu(x)
        x1 = self.pooling(x)
        x2 = self.pooling(x1)
        x3 = self.pooling(x2)
        x4 = self.pooling(x3)
        x5 = self.pooling(x4)
        x6 = self.pooling(x5)
        x7 = self.pooling(x6)
        x8 = self.pooling(x7)
        x9 = self.pooling(x8)
        x10 = self.pooling(x9)
        concated = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], dim=1)
        x = self.concat(concated)
        x = self.pooling(x)
# Inputs to the model
x2 = torch.randn(1, 3, 40, 40)
