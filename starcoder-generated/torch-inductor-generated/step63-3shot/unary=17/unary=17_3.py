
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 16, 3, stride=2)
        self.conv = torch.nn.ConvTranspose2d(16, 1, 5, padding=1)
    def forward(self, x1, x2):
        v2 = self.conv_transpose(x1)
        v1 = self.conv(v2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
