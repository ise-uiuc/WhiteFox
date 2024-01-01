
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 50, kernel_size=(13, 12), stride=(3, 3), padding=(3, 5), output_padding=(1, 2))
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the mode;
x1 = torch.randn(1, 5, 1301, 4096)
