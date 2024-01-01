
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 40, 5, 1)
        self.pool = torch.nn.MaxPool2d(2, 2)
    def forward(self, x):
        self.conv3 = torch.nn.Conv2d(16, 33, 3, 1)
        self.conv4 = torch.nn.Conv2d(33, 66, 3, 1)
        x = x.size(0)
        y = self.conv1(x)
        z = y+1.0
        y = self.conv2(y)  # torch.Size([64, 20, 24, 24])
        y = self.conv3(y)
        y = y - 1
        y = self.conv4(y)
        v1 = z + y   # self.conv3
        v2 = self.pool(v1)  # self.conv4
        return (v2)
# Inputs to the model
x = torch.randn(64, 1, 28, 28)
