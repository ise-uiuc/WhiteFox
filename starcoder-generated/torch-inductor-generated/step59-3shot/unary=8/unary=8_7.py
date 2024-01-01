
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=4, stride=2, padding=1, output_padding=1)
    def forward(self, input1):
        v1 = self.conv(input1)
        v2 = self.conv_transpose(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v2 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
input1 = torch.randn(1, 1, 16, 16, requires_grad=True)
