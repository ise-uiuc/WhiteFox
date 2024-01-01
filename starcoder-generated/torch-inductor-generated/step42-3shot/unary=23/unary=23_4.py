
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(1, 7, 6, stride=3, padding=1, dilation=2, groups=5)
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 55, 22, 18)
