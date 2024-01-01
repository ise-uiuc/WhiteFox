
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(13, 59, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = v1 > 0
        v3 = v1 * 0.362
        v4 = torch.where(v2, v1, v3)
        return v4

# Inputs to the model
x1 = torch.randn(14, 13, 7, 7)
