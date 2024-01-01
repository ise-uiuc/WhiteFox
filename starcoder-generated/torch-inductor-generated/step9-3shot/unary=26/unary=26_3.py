
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(15, 67, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = v1 > 0
        v3 = v1 * 0.98
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(35, 15, 8, 8)
