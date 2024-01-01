
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, out_channels=1, kernel_size=7, stride=3, padding=0)
        self.upsample = torch.nn.Upsample(scale_factor=3.0, mode='bicubic')
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.upsample(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
