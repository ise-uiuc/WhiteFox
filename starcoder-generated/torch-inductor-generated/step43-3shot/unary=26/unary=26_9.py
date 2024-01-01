
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(292, 292, 3, stride=1, padding=1, bias=False)
        self.conv_t2 = torch.nn.ConvTranspose2d(292, 15, 5, stride=2, padding=3, bias=False)
    def forward(self, x8):
        v5 = self.conv_t1(x8)
        v6 = v5 > 0
        v7 = v5 * -1.6052
        v8 = torch.where(v6, v5, v7)
        v9 = self.conv_t2(x8)
        v10 = v9 > 0
        v11 = v9 * -1.1195
        v12 = torch.where(v10, v9, v11)
        return v8 + v12
# Inputs to the model
x8 = torch.randn(24, 292, 77, 93)
