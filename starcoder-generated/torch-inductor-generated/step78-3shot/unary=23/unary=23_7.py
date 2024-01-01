
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 256, kernel_size=(2, 3), stride=(3, 2), padding=(27, 23), output_padding=(1, 32), dilation=(23, 4))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(239, 1, 29, 49)
