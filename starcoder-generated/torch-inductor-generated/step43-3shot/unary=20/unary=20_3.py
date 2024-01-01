
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(96, 64, kernel_size=3, stride=3, padding=0)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.tanh(v1)
        v3 = torch.mul(v2, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 96, 192, 75)
