
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(1, 3, kernel_size=(2, 2, 2), stride=(1, 1, 1), bias=False, output_padding=(0, 0, 0), groups=2, dilation=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(3, 1, 1, 5, 5)
