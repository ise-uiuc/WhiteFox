
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 2016, 4, stride=4, padding=2, groups=2016)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
