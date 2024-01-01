
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(512, 7, 8, bias=True)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * 0.87667
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x = torch.randn(3, 512, 15, 22)
