
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(4, 3, kernel_size=(1, 20), stride=(1, 3), dilation=(1, 1), padding=(0, 0), groups=1)
    def forward(self, x1):
        v1 = self.conv_transpose_2(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(5, 4, 8, 20)
