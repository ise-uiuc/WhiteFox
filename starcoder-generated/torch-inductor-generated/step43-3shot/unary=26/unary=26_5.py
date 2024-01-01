
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(35, 48, 4, stride=1, padding=0, bias=True)
    def forward(self, x12):
        v5 = self.conv_t(x12)
        v6 = v5 > 0
        v7 = v5 * 1.9572
        v8 = torch.where(v6, v5, v7)
        return v8
# Inputs to the model
x12 = torch.randn(48, 35, 31, 15)
