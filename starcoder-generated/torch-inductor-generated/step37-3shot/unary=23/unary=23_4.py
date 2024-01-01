
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 3, 7, stride=2, dilation=1, padding=0)
    def forward(self, x1):
        v1 = torch.tanh(self.conv_transpose(x1))
        return v1

# This pattern is generated as a subpart of the new pattern
x0, x1 = torch.randn(1, 2, 8, 8), torch.randn(2)
