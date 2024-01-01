
class Model(torch.nn.Module):
    def __init__(self, p0):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 64, kernel_size=1, padding=1, stride=1, output_padding=1)
        self.softmax = torch.nn.Softmax(dim=p0+1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return self.softmax(v6)
# Inputs to the model
x1 = torch.randn(1, 2, 66, 66)
