
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t_r = torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=2)
    def forward(self, x1):
        v1 = self.conv_t_r(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 128, 16, 16)
