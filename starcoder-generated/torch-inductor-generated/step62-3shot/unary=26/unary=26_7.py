
class Model(torch.nn.Module):
    def __init__(self, padding1, padding2):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(2, 1, 7)
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 2, 2, stride=2, padding=padding1)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(2, 2, 2, stride=2, padding=padding2, output_padding=1)
    def forward(self, x):
        x_conv = self.conv(x)
        h0 = self.conv_transpose(x_conv)
        h1 = self.conv_transpose1(x_conv)
        return x, h0, h1
padding1 = 1
padding2 = 2
# Inputs to the model
x = torch.randn(1, 2, 10, 10)
