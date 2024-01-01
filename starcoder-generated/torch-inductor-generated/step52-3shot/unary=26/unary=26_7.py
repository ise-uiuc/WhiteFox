
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(95, 95, 7, bias=False)
    def forward(self, x7):
        v5 = self.conv_t(x7)
        v6 = v5 > 0
        v7 = v5 * -1.5
        v8 = torch.where(v6, v5, v7)
        return v8
# Inputs to the model
x7 = torch.randn(19, 94, 10, 12)
