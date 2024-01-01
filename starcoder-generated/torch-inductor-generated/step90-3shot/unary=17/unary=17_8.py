
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 2, 5, 2)
        self.conv2 = torch.nn.Conv2d(2, 1, 3, 1, 1)
        self.bn2d = torch.nn.BatchNorm2d(1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        v3 = self.bn2d(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
