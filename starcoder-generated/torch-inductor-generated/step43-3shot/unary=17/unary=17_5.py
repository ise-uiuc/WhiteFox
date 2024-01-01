
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 128, 3, stride=2, kernel_size=3, padding=0, output_padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
