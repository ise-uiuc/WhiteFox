
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        x2 = self.conv_transpose(x1)
        x3 = x2 > 0
        x4 = x2 * 1
        x5 = torch.where(x3, x2, x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
