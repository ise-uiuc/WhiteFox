
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(94, 83, (5, 2, 7), stride=(1, 2, 3), padding=(2, 0, 4), output_padding=(0, 1, 2), bias=False)
    def forward(self, x8):
        v1 = self.conv_t(x8)
        v2 = v1 > 0
        v3 = v1 * 0.382
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x8 = torch.randn(9, 94, 20, 60, 98)
