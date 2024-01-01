
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v2 = self.conv_transpose(x1)
        v1 = v2 > 0
        v3 = v2 * 2
        v4 = torch.where(v1, v2, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
