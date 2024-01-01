
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channels = 32
        self.conv1 = torch.nn.Conv2d(in_channels=32, out_channels=self.out_channels, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
