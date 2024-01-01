
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(54, 62, 13, 9, 107)
        # self.conv_transpose_pad = torch.nn.ZeroPad2d(int(5432 - 2 * round((5432 - (13 - 1) * 9 - 107) / 2))) # Equivalent to the one-liner below:
        # self.conv_transpose_pad = torch.nn.ZeroPad2d(padding=torch.nn.ZeroPad2d.convert_padding([[int(5432 - 2 * round((5432 - (13 - 1) * 9 - 107) / 2))]]))
        # padding=torch.nn.ZeroPad2d.convert_padding([[int()]])) is the old version, the new version is padding=torch.nn.ZeroPad2d.calculate_padding(input=torch.nn.ConvTranspose2d(x, channels, kernel_size, stride, output_padding)) https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d
    def forward(self, x1):
        # v1 = self.conv_transpose(x1)
        # v2 = v1 + 3
        # v3 = torch.clamp(v2, min=0)
        # v4 = torch.clamp(v3, max=6)
        # v5 = v1 * v4
        # v6 = v5 / 6
        v1 = self.conv_transpose_pad(x1) # Equivalent to v1 = self.conv_transpose(self.conv_transpose_pad(x1))
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 54, 111, 233)
