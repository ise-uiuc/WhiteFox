
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 1, stride=1)
        self.conv_t = torch.nn.ConvTranspose2d(2, 5, 2, stride=1, padding=1)
    def forward(self, x):
        v0 = self.conv(x)
        v1 = self.conv_t(v0)
        v2 = v1 > 0
        v3 = v1 * 0.718
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x = torch.randn(2, 1, 2, 2)
