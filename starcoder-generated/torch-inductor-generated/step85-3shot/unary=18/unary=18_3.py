
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=7, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
