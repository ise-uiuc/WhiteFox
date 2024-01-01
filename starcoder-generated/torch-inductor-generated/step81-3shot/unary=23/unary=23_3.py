
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 2, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(2, 1, kernel_size=3, stride=1, padding=0, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 100, 100)
