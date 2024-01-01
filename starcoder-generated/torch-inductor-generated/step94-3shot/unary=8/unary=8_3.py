
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(6, 9, kernel_size=(2, 2), stride=1, padding=2, output_padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose3d(6, 9, kernel_size=(2, 2, 2), stride=1, padding=2, output_padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(x1)
        v3 = v1 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v1 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(11, 128, 6, 6)
