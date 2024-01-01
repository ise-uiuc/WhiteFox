
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 2, kernel_size=2, stride=2, bias=True, padding=(1, 1))
    def forward(self, inp):
        v1 = self.conv_transpose(inp)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
inp = torch.randn(10, 1, 32, 32)
