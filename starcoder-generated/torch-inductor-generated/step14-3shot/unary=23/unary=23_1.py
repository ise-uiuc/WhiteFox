
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=2, stride=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 2, kernel_size=5, stride=2, padding=1, dilation=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
