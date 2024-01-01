
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(kernel_size=7, in_channels=64, out_channels=64, stride=2)
        self.conv2 = torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, stride=2)
        self.conv3 = torch.nn.Conv2d(kernel_size=5, in_channels=64, out_channels=32, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 64, 128, 128)
