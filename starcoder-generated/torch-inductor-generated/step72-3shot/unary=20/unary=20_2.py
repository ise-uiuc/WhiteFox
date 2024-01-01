
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(64, 256, kernel_size=(2, 5), stride=(1, 2), padding=(1, 2), dilation=2, output_padding=(0, 1))
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 64, 4, 4)
