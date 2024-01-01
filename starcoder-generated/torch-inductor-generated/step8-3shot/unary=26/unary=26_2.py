
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x0):
        x1 = self.conv_transpose(x0)
        x2 = x1 > 0
        x3 = x1 * 0.4
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x0 = torch.randn(1, 1, 5, 5)
