
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(v1+v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
x2 = torch.randn(1, 3, 256, 256)
