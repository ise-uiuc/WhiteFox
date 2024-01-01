
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(7, 5, kernel_size=(2, 19), input_offset=(0, 17), output_padding=(3, 8), bias=True, dilation=(6, 3))
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 7, 19, 29)
