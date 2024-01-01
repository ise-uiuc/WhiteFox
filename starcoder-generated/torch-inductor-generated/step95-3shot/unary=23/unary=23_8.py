
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(10, 9, 11, stride=9, padding=15, dilation=19)
    def forward(self, x1):
        v1 = (self).conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(6, 10, 6, 6)
