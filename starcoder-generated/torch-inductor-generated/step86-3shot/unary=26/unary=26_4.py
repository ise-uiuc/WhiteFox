
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2574, 6, 3, stride=2, padding=1, output_padding=1, bias=False)
    def forward(self, x38):
        v1 = self.conv_t(x38)
        v2 = v1 > 0
        v3 = v1 * -4.42
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x38 = torch.randn(2, 2574, 32, 25)
