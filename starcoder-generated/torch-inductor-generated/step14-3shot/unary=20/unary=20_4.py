
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t4 = torch.nn.ConvTranspose2d(in_channel=2, out_channel=16, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(16)
    def forward(self, x1):
        v1 = self.conv_t4(x1)
        v2 = self.bn(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 16, 16)
