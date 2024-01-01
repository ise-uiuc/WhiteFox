
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(129, 1, 1, stride=1, padding=0, bias=False)
    def forward(self, x0):
        x1 = x0.permute(0, 3, 2, 1)
        x2 = self.conv_t(x1)
        x3 = torch.flatten(x2, 1)
        return x3
# Inputs to the model
x0 = torch.randn(1, 129, 24, 16)
