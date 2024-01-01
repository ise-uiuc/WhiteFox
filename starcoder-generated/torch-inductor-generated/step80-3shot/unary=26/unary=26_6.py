
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(14, 48, 5, stride=1, padding=0)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * -0.9
        x4 = torch.where(x2, x1, x3)
        return x4.numel()
# Inputs to the model
x = torch.randn(14, 14, 12, 6)
