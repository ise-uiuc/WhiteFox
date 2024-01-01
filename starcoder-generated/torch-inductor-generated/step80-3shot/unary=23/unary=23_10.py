
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(3, 2, 3, stride=(1, 1, 1), padding=(0, 1, 1), output_padding=(2, 2, 2), groups=2, dilation=(3, 3, 3), bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 8, 8, 8)
