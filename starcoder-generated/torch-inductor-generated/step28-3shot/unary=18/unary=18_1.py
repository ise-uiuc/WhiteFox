
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=10, stride=10)
    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
