
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, (3, 3))
        self.conv1.bias = torch.nn.Parameter(torch.ones([1], dtype=torch.float32))
        self.conv2 = torch.nn.Conv2d(1, 1, (3, 3))
        self.conv2.bias = torch.nn.Parameter(torch.ones([1], dtype=torch.float32))
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.bn2 = torch.nn.BatchNorm2d(1)

    def forward(self, x):
        x1 = self.conv1(x)
        y = self.conv2(self.bn1(x1))
        return self.bn2(y)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1, 6, 6, dtype=torch.float32)
