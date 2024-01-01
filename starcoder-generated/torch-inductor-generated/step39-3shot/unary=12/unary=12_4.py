
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gconv1 = GatedConvLayer(8, 8, 1, 1)
        self.gconv2 = GatedConvLayer(8, 8, 3, 1)
        self.conv1x1 = torch.nn.Conv2d(8, 12, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.gconv1(x1)
        v2 = self.gconv2(v1)
        v3 = self.conv1x1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
