
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.upool = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=352, out_channels=32, kernel_size=1, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = self.pool(x1)
        v2 = torch.cat((v1, x2), 1)
        v3 = self.upool(v2)
        v4 = self.conv1(v3)
        v5 = torch.sigmoid(v4)
        v6 = self.pool(v5)
        v7 = self.conv2(v6)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
x2 = torch.randn(1, 48, 96, 96)
