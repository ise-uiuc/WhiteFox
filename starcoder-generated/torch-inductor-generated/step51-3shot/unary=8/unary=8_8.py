
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(32, 15, 3, stride=2, padding=1, output_padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(15, 15, 3, stride=2, padding=1, output_padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(15, 32, 3, stride=1, padding=0, output_padding=0)
    def forward(self, x1):
        x2 = self.conv_transpose1(x1)
        x3 = self.conv_transpose2(x2)
        x4 = self.conv_transpose3(x3)
        x5 = x1 + x4
        v1 = x5 + 3
        v2 = torch.clamp(v1, min=0)
        v3 = torch.clamp(v2, max=6)
        v4 = x5 * v3
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 32, 35, 35)
