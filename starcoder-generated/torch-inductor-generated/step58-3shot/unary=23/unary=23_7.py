
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 2, 3, stride=2, padding=1, groups=1, dilation=1)
    def forward(self, x2):
        v1 = self.conv_transpose(x2)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x2 = torch.randn(1, 1, 66, 66)
