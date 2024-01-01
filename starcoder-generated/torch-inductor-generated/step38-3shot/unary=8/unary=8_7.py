
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 1, stride=1, padding=0, output_padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=-0.004075445571706772)
        v4 = torch.clamp(v3, max=1.4908863923072815)
        v5 = v1 * v4
        v6 = v5 / 1.5185217800140381
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
