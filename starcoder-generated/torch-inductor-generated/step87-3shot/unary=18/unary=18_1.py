
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3)
        self.conv2 = torch.nn.Conv2d(8, 8, 3)
        self.conv3 = torch.nn.Conv2d(8, 1, 3)
        self.pool = torch.nn.AvgPool2d(kernel_size=2)
        self.flatten = Flatten()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.pool(v6)
        return self.flatten(v7)
# Inputs to the model
x1 = torch.randn(1, 3, 227, 227)
