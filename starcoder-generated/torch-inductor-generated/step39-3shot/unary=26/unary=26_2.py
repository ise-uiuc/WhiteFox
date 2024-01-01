
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(14, 255, 3, stride=1, padding=1, bias=False)
    def forward(self, x2):
        x1 = torch.abs(self.conv_t(x2) * 0.267 * -10.882 - 5.347 * 0.017)
        return x1
# Inputs to the model
x2 = torch.randn(1, 14, 50, 50)
