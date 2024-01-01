
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(6, 39, stride=1, bias=False, kernel_size=336, padding=62)
    def forward(self, x292):
        x1 = self.conv_t(x292)
        x2 = x1 > 0
        x3 = x1 * -0.435
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x292 = torch.randn(6, 6, 47)
