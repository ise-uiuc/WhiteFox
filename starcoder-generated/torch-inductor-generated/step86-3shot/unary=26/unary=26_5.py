
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(34, 63, 3, stride=2, padding=1, output_padding=1, bias=False)
    def forward(self, x33):
        v1 = self.conv_t(x33)
        v2 = v1 > 0
        v3 = v1 * -4.46
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x33 = -torch.randint(300, 500, (1, 34))
