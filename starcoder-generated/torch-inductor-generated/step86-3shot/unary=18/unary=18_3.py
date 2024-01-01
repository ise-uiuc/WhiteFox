
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)
        self.upsampling = torch.nn.Upsample(scale_factor=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.upsampling(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
