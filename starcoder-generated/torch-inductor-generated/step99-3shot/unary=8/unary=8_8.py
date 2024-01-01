
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose3d(4, 1, 3, stride=(1, 1, 2), dilation=(1, 1, 1),
                                        output_padding=(0, 0, 1), groups=1, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose3d(1, 4, 3, stride=(2, 3, 3), dilation=(2, 1, 2),
                                        output_padding=(1, 2, 1), groups=1, padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose3d(4, 1, 3, stride=(1, 1, 2), dilation=(1, 2, 1),
                                        output_padding=(0, 1, 1), groups=1, padding=1)
        self.conv_transpose4 = torch.nn.ConvTranspose3d(1, 6, 3, stride=(2, 3, 3), dilation=(2, 2, 3),
                                        output_padding=(2, 1, 2), groups=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(2, 4, 24, 32, 16)
