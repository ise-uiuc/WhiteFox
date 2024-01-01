
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        self.conv1 = torch.nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=3, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
