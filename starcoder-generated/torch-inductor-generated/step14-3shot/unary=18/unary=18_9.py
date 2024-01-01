
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=15, stride=5, padding=10)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
