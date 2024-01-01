
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=8, out_channels=12, kernel_size=1, stride=1, padding=1)

    def forward(self, input):
        v1 = torch.nn.functional.interpolate(input, scale_factor=1/5, mode='bicubic')
        v2 = self.conv(v1)
        v3 = torch.sigmoid(v2)
        v4 = torch.nn.functional.interpolate(v3, scale_factor=5, mode='bicubic')
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)
