
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1, stride=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 16, 7, stride=7)
    def forward(self, input):
        v1 = self.conv(input)
        v2 = self.conv_transpose(v1)
        return v2
# Inputs to the model
input = torch.randn(1, 3, 64, 64)
