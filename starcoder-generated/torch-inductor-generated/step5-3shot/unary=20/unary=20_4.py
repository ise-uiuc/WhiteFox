
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), dilation=(2, 3), groups=1, output_padding=(2, 2))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 73, 97)
