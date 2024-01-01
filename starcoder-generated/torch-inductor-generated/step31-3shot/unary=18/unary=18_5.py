
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)
    def forward(self, x3):
        v1 = self.conv1(x3)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x3 = torch.randn(1, 1, 64, 64)
