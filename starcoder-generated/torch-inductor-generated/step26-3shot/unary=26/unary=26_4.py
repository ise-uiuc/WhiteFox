
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(125, 631, 5, stride=3, padding=4, output_padding=2)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = v1 > 7
        v3 = v1 * 0.91
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.full((10, 125, 11, 11), -1, dtype=torch.float)
