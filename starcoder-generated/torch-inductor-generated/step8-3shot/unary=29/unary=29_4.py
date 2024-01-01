
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 8, 3, stride=1, padding=0)
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        v1 = self.gelu(self.conv_transpose(x1))
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
