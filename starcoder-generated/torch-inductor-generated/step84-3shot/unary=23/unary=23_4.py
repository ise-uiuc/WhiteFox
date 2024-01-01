
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 6, 7, stride=1, padding=0, dilation=3, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.hardtanh(v1, -1, 1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)
