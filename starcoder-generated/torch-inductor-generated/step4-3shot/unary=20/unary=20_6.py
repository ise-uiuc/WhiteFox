
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=4, out_channels=2, kernel_size=9, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 8, kernel_size=9, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True)
    def forward(self, X):
        v1 = self.conv(X)
        v2 = self.conv_transpose(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
