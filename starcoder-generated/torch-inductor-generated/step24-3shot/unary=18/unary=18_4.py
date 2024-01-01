
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=16, out_channels=35, kernel_size=7, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=35, out_channels=128, kernel_size=7, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=233, kernel_size=7, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 256, 256)
