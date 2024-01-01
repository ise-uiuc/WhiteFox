
# conv_transpose with valid padding
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_t = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(5, 4), stride=(3, 9), padding=(1, 3), output_padding=2, dilation=1, groups=1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 301, 604)
