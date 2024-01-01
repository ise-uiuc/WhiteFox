
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 31, 9, stride=1, padding=0, output_padding=0)
        self.gelu = torch.nn.GELU()
    def forward(self, x):
        v1 = self.conv_transpose(x)
        v2 = self.gelu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
