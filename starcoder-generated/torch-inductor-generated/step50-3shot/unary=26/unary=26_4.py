
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 200, 5, stride=2, padding=2, output_padding=1, bias=False)
    def forward(self, x4):
        v1 = self.conv_transpose(x4)
        v2 = v1 > 0
        v3 = v1 * 0.175
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x4 = torch.randn(3, 16, 47, 34)
