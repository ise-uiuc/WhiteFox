
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(6, 6, kernel_size=(5, 5), stride=(1, 1), padding=(4, 4), dilation=(2, 2), groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
