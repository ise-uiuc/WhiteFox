
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=3)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=4)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
