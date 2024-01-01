
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(27, 27, (7, 5), stride=(2, 1), padding=(1, 2), output_padding=(1, 3), dilation=(2, 4))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 27, 2, 4)
