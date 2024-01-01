
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=3)
        self.pool1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.pool1(v1)
        v3 = self.conv2(v2)
        v4 = v3 + 9
        return v4
# Inputs to the model
x = torch.randn(1, 3, 256, 256)
