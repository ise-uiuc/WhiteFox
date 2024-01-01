
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 32, 3, stride=2, padding=1, dilation=1, output_padding=0)
        self.conv = torch.nn.Conv2d(32, 32, 3, stride=1, padding=3, dilation=1, output_padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv(v1)
        v1 = torch.relu(v2)
        v3 = v1 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
