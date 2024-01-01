
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(35, 1, 4, stride=1, padding=1, bias=False)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = v1 > 0
        v4 = torch.where(v2, v1, v1)
        return v4
# Inputs to the model
x1 = torch.randn(2, 35, 15, 27)
