
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(17, 4, 1, stride=2)
        self.pointwise_convolution = torch.nn.Conv2d(in_channels=17, out_channels=4, kernel_size=(1, 1), stride=(1, 1))
        self.affine_transform_constant = 3
        self.clamp_max = 6
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.pointwise_convolution(x1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0, max=6)
        v5 = v2 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 17, 32, 32)
