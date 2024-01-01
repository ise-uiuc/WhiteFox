
class Model(torch.nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 2, 3, stride=2, padding=1, bias=bias)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 7, 7)
