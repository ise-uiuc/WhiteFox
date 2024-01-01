
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_transpose = torch.nn.ConvTranspose2d(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=1, bias=False)
    def forward(self, x1):
        v1 = self.conv1_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(2, 2, 8, 8)
