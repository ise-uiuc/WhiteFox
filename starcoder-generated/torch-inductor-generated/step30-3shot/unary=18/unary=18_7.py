
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Depthwise convolution without BN or activation
        self.conv1 = torch.nn.Conv2d(32, 32, kernel_size=(14, 17), stride=(12, 24), padding=(8, 10), groups=32)
        self.conv2 = torch.nn.Conv2d(32, 48, kernel_size=(10, 7), stride=(11, 13), padding=(7, 3))
        self.conv3 = torch.nn.Conv2d(48, 96, kernel_size=(10, 10), stride=(9, 10), padding=(7, 6))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = torch.sigmoid(v2)
        v6 = self.conv3(v4 + v5)
        v7 = torch.sigmoid(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 32, 152, 212)
