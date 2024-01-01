
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 7x7 square convolution
        # kernel
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 32, 7)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 64, 128, 128)
