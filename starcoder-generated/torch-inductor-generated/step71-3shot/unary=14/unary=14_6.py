
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1_2 = torch.nn.ConvTranspose2d(37, 49, 1, stride=2, padding=0, output_padding=0)
        self.conv_transpose_2_2 = torch.nn.ConvTranspose2d(49, 60, 4, stride=2, padding=0, output_padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_1_2(x1)
        v2 = self.conv_transpose_2_2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 37, 26, 26)
