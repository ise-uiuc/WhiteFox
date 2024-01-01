
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(17, 19), stride=3, padding=4, dilation=0, output_padding=1, groups=6, bias=True)
    def forward(self, x1):
        v1 = torch.sigmoid(self.conv_transpose(x1))
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
