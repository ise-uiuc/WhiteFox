
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 6, stride=2, padding=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 4, 5, stride=1, padding=1, output_padding=1)
        self.conv_1x1 = torch.nn.Conv2d(4, 8, 1)
        self.conv_transpose_1x1 = torch.nn.ConvTranspose2d(8, 4, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.relu(v2)
        v4 = self.conv_1x1(v3)
        v5 = torch.hardtanh(v4)
        v6 = self.conv_transpose_1x1(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
