
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=2, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 8, kernel_size=2, stride=1, dilation=1, groups=1, bias=True)
    def forward(self, x1):
        v1 = F.relu(x1)
        v2 = self.conv(v1)
        v3 = F.relu(v2)
        v4 = self.conv_transpose(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
