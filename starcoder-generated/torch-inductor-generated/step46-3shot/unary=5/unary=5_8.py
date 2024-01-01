
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 1, stride=2, padding=1, dilation=1, bias=False, groups=1)
        self.batch_norm = torch.nn.BatchNorm2d(1)
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 3, stride=1, padding=1, dilation=1, output_padding=1, groups=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.batch_norm(v1)
        v3 = self.conv(v2)
        v4 = self.conv_transpose(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
