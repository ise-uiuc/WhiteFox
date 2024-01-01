
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(in_channels=1, out_channels=2, kernel_size=(5, 3), stride=(3, 1), padding=(2, 1), output_padding=(0, 0), bias=None)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
